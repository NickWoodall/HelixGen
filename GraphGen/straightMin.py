
import sys
sys.path.insert(0,f'../')

import pyrosetta.distributed.tasks.score as score
import pyrosetta


from pyrosetta.rosetta.utility import vector1_bool
from pyrosetta.rosetta.core.pose import PDBInfo
from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code 
from pyrosetta.rosetta.protocols import rosetta_scripts
import pyrosetta.distributed.packed_pose as packed_pose




from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,copysign,atan2
import math
import random


import HelixFit as hf
import util.RotationMethods as rm


import argparse
import os


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


def gen_coord_data(prot):
    coord_res = dict()
    
    for x in range(len(prot.helixRes)):
        caList = prot.get_CA_list_fit_()
        for y in range(1,len(prot.helixRes[x])-1):
            coord_res[prot.helixRes[x][y]] =  caList[x][y]
            
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


def str_min(inDire,filename,oD):
    
    
    pose = pyrosetta.pose_from_pdb(f'{inDire}/{filename}')

    prot = hf.HelicalProtein(filename,direc=f'{inDire}/',name=filename[:-4])
    prot.fit_all()

    d = gen_coord_data(prot)

    mutSeq = 'A'*len(pose.sequence())
    pose = mutate_residue(pose, mutSeq)
    pyrosetta.rosetta.core.pose.addVirtualResAsRoot(pose)
    
    for key in d.keys():
        pose.add_constraint(genCoordConstraint(pose,d,key))

    pose = straight_min(pose)
    pose.dump_pdb(f'{oD}/{filename[:-4]}_str.pdb')



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--outDirec', help='Place to put straightened proteins.', default = 'output/')
    parser.add_argument('-i','--inDirec', help='Protein directory to fit and straighten', default = '../data/BCov_4H_dataset/BCov_Models')
    parser.add_argument('-j','--jran', help='Constant Seed to Initialize pyrosetta with', default = 1111111, type=int)
    args = parser.parse_args()
    #pyrosetta.init(f'-beta -constant_seed True -jran {args.jran}')
    
    
    fileList = os.listdir(args.inDirec)

    for x in fileList:
        if not x.endswith('.pdb'):
            continue
        #protect against not 4 helices detected; ignore these
        try:
            randNum = random.randrange(0, 9999999, 1)
            pyrosetta.init(f'-beta -constant_seed True -jran {randNum}')
            str_min(args.inDirec,x,args.outDirec)
        except:
            print(f'Error loading {x}')

    
    

    

