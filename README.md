# BAP
Binding Affinity Prediction for Protein-ligand Complexes
#############################################################################
#                NOTEs for the BAP repository                               #
#############################################################################

1. Requirements
   BAP currently supports a Linux system and Python 3.7, and requires main dependency packages as follows. 
   - deemchem (https://github.com/deepchem/deepchem)
   - rdkit (https://www.rdkit.org/)
   - numpy (https://numpy.org/)		
   - pandas (https://pandas.pydata.org/)
   - sklearn (https://scikit-learn.org/stable/)
   - tensorflow (https://www.tensorflow.org/)
   - multiprocessing (https://docs.python.org/3/library/multiprocessing.html)
   - biopandas (http://rasbt.github.io/biopandas/)
   - scipy (https://www.scipy.org/)

2. Data downloading and preprocessing
	1) Downloading:
	   Model construction - 'refined-set' data folder: PDBbind refined set v2020 (http://www.pdbbind.org.cn/)
	   validation - 'casf2016' data folder: CASF-2016 set in PDBbind (http://www.pdbbind.org.cn/)
				  - 'csarhiqS1' data folder: CSAR-HiQ sets 1 (http://www.csardock.org/)
				  - 'csarhiqS2' data folder: CSAR-HiQ sets 2 (http://www.csardock.org/)
				  - 'csarhiqS3' data folder: CSAR-HiQ sets 3 (http://www.csardock.org/)

	2) Preprocessing:
	   'refined-set': save the ligand files as PDB files and the protein files as MOL2 files (e.g. using software like UCSF Chimera)
	   'casf2016': save the ligand files as PDB files and the protein files as MOL2 files (e.g. using software like UCSF Chimera)
	   'csarhiqS1', 'csarhiqS2' and 'csarhiqS3': save the protein and ligand in each complex as PDB files (e.g. using software like UCSF Chimera), 
	   											 and name these files as those in PDBbind refined set (e.g. 1ax1_protein.pdb, 1ax1_ligand.pdb in '1ax1' folder)
	3) Put these folders together:
	   - Create a root folder (e.g. 'PDBbind') 
	   - Create a 'v2020' foler in 'PDBbind', a 'PDBbind_v2020_refined' folder in 'v2020', and put 'refined-set' folder in 'PDBbind_v2020_refined'
	   - Create a 'ValidationSets' folder, and put 'casf2016', 'csarhiqS1', 'csarhiqS2' and 'csarhiqS3' in 'ValidationSets'
	   - Put the 'indexes' folder (provided in this repository) in 'PDBbind'
	   
3. Example codes are provided in the 'Examples' folder in this repository
   1) IMCP-Score - Constructing an IMCP-based machine-learning SF on PDBbind refined set v2020 (excluding the validation sets) and
   				   validating it on the four validation sets (CASF-2016 and CSAR-HiQ sets) using Pearson's correlation and RMSE
   2) IMCPiDB-Score - Constructing an IMCPiDB-based deep-learning SF on PDBbind refined set v2020 (excluding the validation sets) and
   				      validating it on the four validation sets (CASF-2016 and CSAR-HiQ sets) using Pearson's correlation and RMSE
