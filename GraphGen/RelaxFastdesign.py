import os
import pyrosetta
from pyrosetta.rosetta.utility import vector1_bool
from pyrosetta.rosetta.core.pose import PDBInfo
from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code 
from pyrosetta.rosetta.protocols import rosetta_scripts
import pyrosetta.distributed.packed_pose as packed_pose
pyrosetta.init("-beta -mute all")
import os, time, gzip
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from collections import defaultdict
import urllib.request, json 
import time
import timeit
import sys, glob
import shutil
from pymol import cmd, stored, selector
from weakref import WeakKeyDictionary
import pickle


import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

sys.path.insert(0,f'experiments/')
sys.path.insert(0,f'struct2seq/')
from struct2seq import *
from utils import *
import data
import noam_opt

def mutate_residue( pose , newSeq):
    """Taken from gray lab mutateresidue function changes all amino in pose to newSeq with one repack
    """

    test_pose = pyrosetta.pyrosetta.Pose()
    test_pose.assign( pose )

    # create a standard scorefxn by default
    pack_scorefxn = pyrosetta.rosetta.core.scoring.ScoreFunction()

    task = pyrosetta.standard_packer_task(test_pose)
    
    mutant_position = 1
    
    for x in newSeq:

        # the Vector1 of booleans (a specific object) is needed for specifying the
        #    mutation, this demonstrates another more direct method of setting
        #    PackerTask options for design
        aa_bool = vector1_bool()
        # PyRosetta uses several ways of tracking amino acids (ResidueTypes)
        # the numbers 1-20 correspond individually to the 20 proteogenic amino acids
        # aa_from_oneletter returns the integer representation of an amino acid
        #    from its one letter code
        # convert mutant_aa to its integer representation
        mutant_aa = aa_from_oneletter_code(x)

        # mutation is performed by using a PackerTask with only the mutant
        #    amino acid available during design
        # to do this, construct a Vector1 of booleans indicating which amino acid
        #    (by its numerical designation, see above) to allow
        for i in range( 1 , 21 ):
            # in Python, logical expression are evaluated with priority, thus the
            #    line below appends to aa_bool the truth (True or False) of the
            #    statement i == mutant_aa
            aa_bool.append( i == mutant_aa )

        # modify the mutating residue's assignment in the PackerTask using the
        #    Vector1 of booleans across the proteogenic amino acids
        task.nonconst_residue_task(mutant_position).restrict_absent_canonical_aas( aa_bool )
        mutant_position += 1

    # apply the mutation and pack nearby residues
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover( pack_scorefxn , task )
    packer.apply(test_pose)

    return test_pose


def gen_coord_data(prot):
    
    hL = []
    #get residues of helices in protein used for fit, some error in duplication of some data from prot, fix with this loop
    for x in range(len(prot.helix_list)):
        if x-len(prot.helix_list)<-1:
            hL.append(prot.helix_list[x][1:-1].copy())
        else:
            if prot.helix_list[x][0] == prot.helix_list[x][1]:
                hL.append(prot.helix_list[x][1:-1])
    
    ca_coord_array = prot.get_CA_Coords_array()
    coord_res = dict()
    
    for x in range(len(hL)):
        for y in range(len(hL[x])):
            coord_res[hL[x][y]] =  ca_coord_array[x][y]
            
    return coord_res

def rosVec(coord_data,num):
    
    return pyrosetta.rosetta.numeric.xyzVector_double_t(coord_data[num][0],coord_data[num][1],coord_data[num][2])
    
def genCoordConstraint(pose,coord_data,num):
    t1 = pyrosetta.rosetta.core.scoring.func.HarmonicFunc(0,0.1)
    CAnum = pyrosetta.AtomID(pose.residue(num).atom_index("CA"),num)

    length = len(pose.sequence())
    vr = pyrosetta.AtomID(pose.residue(length).atom_index('X'),length)

    coord=rosVec(coord_data,num)

    return pyrosetta.rosetta.core.scoring.constraints.CoordinateConstraint(CAnum,vr,coord,t1)

def fastRelax(input_pose):
    #import pyrosetta.rosetta.core.pose as pose
    work_pose = packed_pose.to_pose(input_pose)
    
    protocol="""<ROSETTASCRIPTS>

	<SCOREFXNS>
		<ScoreFunction name="beta" weights="beta"/>
	</SCOREFXNS>

	<RESIDUE_SELECTORS>
         <Chain name="chA" chains="A"/>
	</RESIDUE_SELECTORS>

 	<TASKOPERATIONS>
		<InitializeFromCommandline name="init"/>
		<IncludeCurrent name="current"/>
		<LimitAromaChi2 name="arochi" />
        <ExtraRotamersGeneric name="ex1_ex2" ex1="1" ex2="1"/> # Add increased sampling for chi1 and chi2 rotamers
        <ExtraRotamersGeneric name="ex1_ex4" ex1="1" ex2="1" ex3="1" ex4="1"/>
        <OperateOnResidueSubset name="re" selector="chA">
			<RestrictToRepackingRLT/>
		</OperateOnResidueSubset>
        
	</TASKOPERATIONS>
 	
	<FILTERS>
 	</FILTERS>

	<MOVERS>
		<FastDesign name="FDE" scorefxn="beta" task_operations="re,ex1_ex2" repeats="1" relaxscript="MonomerRelax2019" >
			<MoveMap name="what_moves" bb="true" chi="true" jump="true" />
		</FastDesign>

		<MinMover name="min_sc" scorefxn="beta" chi="true" bb="false" jump="ALL" cartesian="false" type="dfpmin_armijo_nonmonotone" tolerance="0.0001" max_iter="5000" />
		<MinMover name="min_all" scorefxn="beta" chi="true" bb="true" jump="ALL" cartesian="false" type="dfpmin_armijo_nonmonotone" tolerance="0.0001" max_iter="5000" />
	</MOVERS>

	<PROTOCOLS>
		<Add mover="FDE"/>
        Add mover="min_sc" />
        <Add mover="min_all" />
	</PROTOCOLS>
</ROSETTASCRIPTS>"""
    
    xml = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(protocol)
    protoc = xml.get_mover("ParsedProtocol")
    protoc.apply(work_pose)
    
    return work_pose

    
def fastDesign(input_pose):
    
    work_pose = packed_pose.to_pose(input_pose)

    protocol="""<ROSETTASCRIPTS>

   <SCOREFXNS>
	   
	<ScoreFunction name="score_FDE" weights="beta_cst">
	</ScoreFunction>
	
	<ScoreFunction name="natural" weights="beta"/>
	
	<ScoreFunction name="up_ele" weights="beta"> # for designing surface
        	<Reweight scoretype="fa_elec" weight="1.4"/>
        	<Reweight scoretype="hbond_sc" weight="2.0" />
        	<Reweight scoretype="buried_unsatisfied_penalty" weight="1.0" />
        	<Reweight scoretype="res_type_constraint" weight="2"/> # Must have this in for loop design
    </ScoreFunction>
	
	<ScoreFunction name="cst" weights="beta_cst" />
	

   </SCOREFXNS>


<RESIDUE_SELECTORS>
		<Layer name="surface" select_core="false" select_boundary="false" select_surface="true" use_sidechain_neighbors="true" core_cutoff="3.3" surface_cutoff="2"/> 
		<Layer name="boundary" select_core="false" select_boundary="true" select_surface="false" use_sidechain_neighbors="true" core_cutoff="3.3" surface_cutoff="2"/>
		<Layer name="core" select_core="true" select_boundary="false" select_surface="false" use_sidechain_neighbors="true" core_cutoff="3.3" surface_cutoff="2"/>

        <Not name="not_surface" selector="surface" />
		<Not name="not_boundary" selector="boundary" />
		<Not name="not_core" selector="core" />
		
        
        <SecondaryStructure name="entire_helix" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="H" />
		<SecondaryStructure name="sheet" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="E" />
		<SecondaryStructure name="entire_loop" overlap="0" minH="3" minE="2" include_terminal_loops="true" use_dssp="true" ss="L" />
		
		<And name="helix_cap" selectors="entire_loop"> # Define a helix cap selection for layer design (C-term)
			<PrimarySequenceNeighborhood lower="1" upper="0" selector="entire_helix"/>
		</And>
		<And name="helix_start" selectors="entire_helix"> # Define a helix start selection for layer design (N-term)
			<PrimarySequenceNeighborhood lower="0" upper="1" selector="helix_cap"/>
		</And>
		<And name="helix" selectors="entire_helix"> # Define helix without its caps
			<Not selector="helix_start"/>
		</And>
		<And name="loop" selectors="entire_loop"> # Define loop without helix caps
			<Not selector="helix_cap"/>
		</And>

		<Neighborhood name="around_loop" distance="4.0" selector="entire_loop"/>
		<Not name="not_around_loop" selector="around_loop"/>

		<Chain name="chA" chains="A"/> # Select chain A from input structure

</RESIDUE_SELECTORS>

<TASKOPERATIONS>


		<IncludeCurrent name="current"/> # Tell the packer to also consider the input rotamer.
		<LimitAromaChi2 name="arochi" /> # Prevents use the rotamers of PHE, TYR and HIS that have chi2 far from 90
		<ExtraRotamersGeneric name="ex1_ex2" ex1="1" ex2="1"/> # Add increased sampling for chi1 and chi2 rotamers
		<ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1"/> # Add increased sampling for chi1 and chi2 rotamers
		<ExtraRotamersGeneric name="ex1" ex1="1" ex2="0"/> # Add increased sampling for chi1 and chi2 rotamers
		<DesignRestrictions name="layer_design"> # Define residues that are allowed at the different parts of the protein (based on layers definition)
			<Action selector_logic="surface AND helix_start"		aas="DEPNQS"/>
			<Action selector_logic="surface AND helix"			aas="EHKQR"/>
			<Action selector_logic="surface AND sheet"			aas="EHKNQRST"/>
			<Action selector_logic="surface AND loop"			aas="DEGHKNPQRST"/>
			<Action selector_logic="boundary AND helix_start"		aas="PNQS"/>
			<Action selector_logic="boundary AND helix"			aas="ADEHIKLMNQRSTV"/>
			<Action selector_logic="boundary AND sheet"			aas="DEFHIKLMNQRSTVY"/>
			<Action selector_logic="boundary AND loop"			aas="AFGHIKLMNPQRSTVED"/>#
			<Action selector_logic="core AND helix_start"			aas="PNQS"/>
			<Action selector_logic="core AND helix"				aas="AFILMVWY"/>
			<Action selector_logic="core AND sheet"				aas="FILMVWY"/>
			<Action selector_logic="core AND loop"				aas="AFGILMPVWY"/>
			<Action selector_logic="helix_cap"				aas="KRHQDNS"/>
		</DesignRestrictions>
	
		<ConsensusLoopDesign name="disallow_non_abego_aas"/>
		<OperateOnResidueSubset name="design_surface" selector="not_surface">
			<PreventRepackingRLT/>
		</OperateOnResidueSubset>
		



   </TASKOPERATIONS>

<FILTERS>

</FILTERS>





   <MOVERS>

		<FastDesign name="FDE" scorefxn="score_FDE" task_operations="current,arochi,layer_design,ex1_ex2,disallow_non_abego_aas" repeats="1" relaxscript="MonomerDesign2019" >
			<MoveMap name="what_moves" bb="true" chi="true" jump="true" />
		</FastDesign>

		<MinMover name="min_sc" scorefxn="cst" chi="true" bb="false" jump="ALL" cartesian="false" type="dfpmin_armijo_nonmonotone" tolerance="0.0001" max_iter="200" />
		<MinMover name="min_all" scorefxn="cst" chi="true" bb="true" jump="ALL" cartesian="false" type="dfpmin_armijo_nonmonotone" tolerance="0.0001" max_iter="200" />

		
		
   </MOVERS>
<PROTOCOLS>
		<Add mover_name="FDE"/>
		<Add mover_name="min_sc"/>
		<Add mover_name="min_all"/>
</PROTOCOLS>


	# The scorefunction specified by the OUTPUT tag will be used to score the pose prior to output
	<OUTPUT scorefxn="natural"> 
	</OUTPUT>

</ROSETTASCRIPTS>""" % locals()
    
    
    xml = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(protocol)
    protoc = xml.get_mover("ParsedProtocol")
    protoc.apply(work_pose)
    
    return work_pose
    
def scoreOnly(input_pose):
    #import pyrosetta.rosetta.core.pose as pose
    work_pose = packed_pose.to_pose(input_pose)
    
    protocol="""<ROSETTASCRIPTS>

	<SCOREFXNS>
		<ScoreFunction name="beta" weights="beta"/>
	</SCOREFXNS>

	<RESIDUE_SELECTORS>
	</RESIDUE_SELECTORS>

 	<TASKOPERATIONS>
	</TASKOPERATIONS>
 	
	<FILTERS>
 	</FILTERS>

	<MOVERS>
	</MOVERS>

	<PROTOCOLS>
	</PROTOCOLS>
    <OUTPUT scorefxn="beta" />
</ROSETTASCRIPTS>"""
    

    xml = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(protocol)
    protoc = xml.get_mover("ParsedProtocol")
    protoc.apply(work_pose)
    
    return work_pose

def score_sssc(input_pose):
    #import pyrosetta.rosetta.core.pose as pose
    work_pose = packed_pose.to_pose(input_pose)
    
    protocol="""<ROSETTASCRIPTS>

	<SCOREFXNS>
		<ScoreFunction name="beta" weights="beta"/>
	</SCOREFXNS>

	<RESIDUE_SELECTORS>
	</RESIDUE_SELECTORS>

 	<TASKOPERATIONS>
	</TASKOPERATIONS>
 	
	<FILTERS>
    <SSShapeComplementarity name="sc" loops="0" helices="1" confidence="0" />
 	</FILTERS>

	<MOVERS>
	</MOVERS>

	<PROTOCOLS>
    <Add filter="sc"/>
	</PROTOCOLS>
    <OUTPUT scorefxn="beta" />
</ROSETTASCRIPTS>"""
    

    xml = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(protocol)
    protoc = xml.get_mover("ParsedProtocol")
    protoc.apply(work_pose)
    
    return work_pose




def straight_min(input_pose):
    """Minimize with constraints. Contraints added to pose in other """
    
    work_pose = pyrosetta.pyrosetta.Pose()
    work_pose.assign(input_pose) 
    
    protocol = """<ROSETTASCRIPTS>

   <SCOREFXNS>
	<ScoreFunction name="natural" weights="beta"/>
	<ScoreFunction name="straight" weights="beta_cst"> # for designing surface
        	<Reweight scoretype="hbond_sr_bb" weight="1.0"/>
        	<Reweight scoretype="hbond_lr_bb" weight="1.0" />
            <Reweight scoretype="coordinate_constraint" weight= "10.0"/>
      	</ScoreFunction>
   </SCOREFXNS>
    <RESIDUE_SELECTORS>
    </RESIDUE_SELECTORS>
  <TASKOPERATIONS>
   </TASKOPERATIONS>

    <FILTERS>
    </FILTERS>
   <MOVERS>
		<MinMover name="min_all" scorefxn="straight" chi="true" bb="true" jump="ALL" cartesian="false" type="dfpmin_armijo_nonmonotone" tolerance="0.0001" max_iter="1000" />
   </MOVERS>
    <PROTOCOLS>
		<Add mover_name="min_all"/>
    </PROTOCOLS>
	# The scorefunction specified by the OUTPUT tag will be used to score the pose prior to output
	<OUTPUT scorefxn="natural"> 
	</OUTPUT>
         </ROSETTASCRIPTS>"""
    
    xml = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(protocol)
    protoc = xml.get_mover("ParsedProtocol")
    protoc.apply(work_pose)
    
    return work_pose


def get_pose_scores(input_pose):
    """Convert .scores into dictionary that can be saved without pyrosetta -serialization build"""
    outDict = dict()
    for key,value in input_pose.scores.items():
        outDict[key]=value
    return outDict
    
#Example Code
# sL is output of getSeq
# for x in range(len(sL)):
    
#     pred_seq = sL[x][0]
#     name = sL[x][1]
    
#     #change residues to graphGen prediction
#     pose = pyrosetta.pose_from_file(f'{direName}{name}.pdb')
#     pred_pose = mutate_residue(pose,pred_seq)
    
    
#     #relax using predicted sequence
#     start_pred = time.time()
#     pred_pose_des_relax = fastRelax(pred_pose)
#     end_pred = time.time()
#     pred_time_relax = end_pred - start_pred

    



