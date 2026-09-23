# Parameterising Small Molecules with OpenFF

This is the first of two jupyter notebooks on handling force fields using [Open Force Field's](https://openforcefield.org/) software, and subsequent molecular dynamics and analysis. This notebook describes the parameterisation of small molecules, while the second notebook (`protein_ligand_complex_parameterisation_and_md.ipynb`) will take you through parameterising a protein-ligand complex, running molecular dynamics, and performing some analysis of pose stability and interactions.

### Prerequisites

 - Basic knowledge of Python
 - Basic familiarity with molecular mechanics force fields and molecular dynamics simulations (see talk by Danny Cole)

### The Plan

| Action | Software|
|--|--|
| [Go from SMILES to simulation in a few lines of code](#showcase) | OpenFF Toolkit, OpenFF Interchange, OpenMM
| [Load and inspect a force field](#loading_ff) | OpenFF Toolkit
| [Create a representation of your chemical system](#topology) | OpenFF Toolkit
| [Parameterise your system and run a quick simulation in water](#interchange) | OpenFF Interchange, OpenMM
| [Rapidly assign partial charges with a graph neural network model](#gnn_charges) | OpenFF Toolkit, OpenFF NAGL Models
| [Review what you've learnt](#summary) | 
| [Check out other OpenFF tutorials](#further_materials) | 
 
### Jupyter Cheat Sheet

- To run the currently highlighted cell and move focus to the next cell, hold <kbd>&#x21E7; Shift</kbd> and press <kbd>&#x23ce; Enter</kbd>;
- To run the currently highlighted cell and keep focus in the same cell, hold <kbd>&#x21E7; Ctrl</kbd> and press <kbd>&#x23ce; Enter</kbd>;
- To get help for a specific function, place the cursor within the function's brackets, hold <kbd>&#x21E7; Shift</kbd>, and press <kbd>&#x21E5; Tab</kbd>;

### Acknowledgements

Most of this material was adapted from the [2023 CCPBioSim Workshop Open Force Field Sessions](https://github.com/openforcefield/ccpbiosim-2023?tab=readme-ov-file) created by Matt Thompson and Jeff Wagner.

### Maintainers
 - Finlay Clark -- finlay.clark@newcastle.ac.uk (@fjclark)



<a id="showcase"></a>
## 0. You can go from SMILES to simulation in a few lines of code


```python
# Go from SMILES -> simulation input with OpenFF
from openff.toolkit import ForceField, Molecule, Topology

molecule = Molecule.from_smiles("CC(=O)Nc1ccc(cc1)O")
molecule.generate_conformers(n_conformers=1)
topology = Topology.from_molecules([molecule])

force_field = ForceField("openff-2.3.0.offxml")
interchange = force_field.create_interchange(topology)
interchange.minimize()

openmm_system = interchange.to_openmm_system()
openmm_topology = interchange.to_openmm_topology()
openmm_positions = interchange.positions.to_openmm()
```


```python
# Run the simulation with OpenMM
import openmm
import openmm.app

temperature = 298.15 * openmm.unit.kelvin
friction_coefficient = 1.0 / openmm.unit.picosecond
step_size = 2.0 * openmm.unit.femtosecond

simulation = openmm.app.Simulation(
    openmm_topology,
    openmm_system,
    openmm.LangevinIntegrator(temperature, friction_coefficient, step_size),
)
simulation.context.setPositions(openmm_positions)
simulation.context.setVelocitiesToTemperature(simulation.integrator.getTemperature())

simulation.reporters.append(
    openmm.app.DCDReporter(file="trajectory_showcase.dcd", reportInterval=100)
)
simulation.step(10000)
```


```python
# Load the trajectory with MDAnalysis and visualise with nglview
import MDAnalysis as mda
import nglview

u = mda.Universe(openmm_topology, "trajectory_showcase.dcd")

view = nglview.show_mdanalysis(u)
view
```


    


    /opt/conda/envs/openff-env/lib/python3.14/site-packages/MDAnalysis/coordinates/DCD.py:171: DeprecationWarning: DCDReader currently makes independent timesteps by copying self.ts while other readers update self.ts inplace. This behavior will be changed in 3.0 to be the same as other readers. Read more at https://github.com/MDAnalysis/mdanalysis/issues/3889 to learn if this change in behavior might affect you.
      warnings.warn("DCDReader currently makes independent timesteps"



    NGLWidget(max_frame=99)


That's it! You've run a vacuum simulation for paracetamol. Below and in the next notebook, we'll go into more detail on each of the steps in the OpenFF cell and show how you can set up more complex systems, but this is mainly for your understanding and you rarely need much more code than shown above.

<a id="loading_ff"></a>
## 1. Force fields are specified in `.offxml` files and can be loaded with the `ForceField` class

Let's dive into the details of what went on above. Here's a summary of how data flows through a workflow utilising OpenFF tools -- the OpenFF toolkit allows you to create `Molecule` and `ForceField` objects, which get combined into an `Interchange` object, which contains all the information needed to start a simulation. From there, you can create input for the simulation engine of your choice:

<img src="../images/openff_flowchart.png" alt="Description of image" style="max-width: 1000px; display: block; margin-left: auto; margin-right: auto;" />

Let's start with the `.offxml` force field file. OpenFF's force fields use the SMIRKS Native Open Force Field (SMIRNOFF) [specification](https://openforcefield.github.io/standards/standards/smirnoff/). The spec fully describes the contents of a SMIRNOFF force field, how parameters should be applied, and several other important usage details. You could implement a SMIRNOFF engine in your own code, but conveniently the OpenFF Toolkit already provides this and a handful of utilities. Let's load up the latest OpenFF small molecule force field, OpenFF 2.3.0, and inspect its contents! This force field shares the code name "Sage" with all other force fields with the same major version number (2.x.x).


```python
from openff.toolkit import ForceField

sage = ForceField("openff-2.3.0.offxml")
sage
```




    <openff.toolkit.typing.engines.smirnoff.forcefield.ForceField at 0x7ff591bf1e50>



If you'd like to see the raw file on disk that's being parsed, [here's the file on GitHub](https://github.com/openforcefield/openff-forcefields/blob/main/openforcefields/offxml/openff-2.3.0.offxml).

Each section of a force field is stored in memory within `ParameterHandler` objects, which can be looked up with brackets (just like looking up values in a dictionary):


```python
print(sage.registered_parameter_handlers)

vdw_handler = sage["vdW"]
vdw_handler
```

    ['Constraints', 'Bonds', 'Angles', 'ProperTorsions', 'ImproperTorsions', 'vdW', 'Electrostatics', 'LibraryCharges', 'NAGLCharges']





    <openff.toolkit.typing.engines.smirnoff.parameters.vdWHandler at 0x7ff591bf2710>



Each `ParameterHandler` in turn stores a list of parameters in its `.parameters` attribute, in addition to some information specific to its portion of the potential energy function:


```python
print(f"vdw_handler cutoff: {vdw_handler.cutoff}")
print(f"vdw_handler combining rules: {vdw_handler.combining_rules}")
print(f"vdw_handler scale14: {vdw_handler.scale14}")
print(f"vdw_handler parameters: {vdw_handler.parameters}")
```

    vdw_handler cutoff: 9.0 angstrom
    vdw_handler combining rules: Lorentz-Berthelot
    vdw_handler scale14: 0.5
    vdw_handler parameters: [<vdWType with smirks: [#1:1]  epsilon: 0.0157 kilocalorie / mole  id: n1  rmin_half: 0.6 angstrom  >, <vdWType with smirks: [#1:1]-[#6X4]  epsilon: 0.01336628116185 kilocalorie / mole  id: n2  rmin_half: 1.495082464255 angstrom  >, <vdWType with smirks: [#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]  epsilon: 0.01891997418601 kilocalorie / mole  id: n3  rmin_half: 1.435967812686 angstrom  >, <vdWType with smirks: [#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]  epsilon: 0.01559137568183 kilocalorie / mole  id: n4  rmin_half: 1.288149753875 angstrom  >, <vdWType with smirks: [#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]  epsilon: 0.01517383637638 kilocalorie / mole  id: n5  rmin_half: 1.188911001242 angstrom  >, <vdWType with smirks: [#1:1]-[#6X4]~[*+1,*+2]  epsilon: 0.0157 kilocalorie / mole  id: n6  rmin_half: 1.1 angstrom  >, <vdWType with smirks: [#1:1]-[#6X3]  epsilon: 0.01597537378736 kilocalorie / mole  id: n7  rmin_half: 1.479065946749 angstrom  >, <vdWType with smirks: [#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]  epsilon: 0.01761379732429 kilocalorie / mole  id: n8  rmin_half: 1.370805406499 angstrom  >, <vdWType with smirks: [#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]  epsilon: 0.01359601107204 kilocalorie / mole  id: n9  rmin_half: 1.372163754664 angstrom  >, <vdWType with smirks: [#1:1]-[#6X2]  epsilon: 0.015 kilocalorie / mole  id: n10  rmin_half: 1.459 angstrom  >, <vdWType with smirks: [#1:1]-[#7]  epsilon: 0.01386809433135 kilocalorie / mole  id: n11  rmin_half: 0.6506218845032 angstrom  >, <vdWType with smirks: [#1:1]-[#8]  epsilon: 1.232058709465e-05 kilocalorie / mole  id: n12  rmin_half: 0.2991902460601 angstrom  >, <vdWType with smirks: [#1:1]-[#16]  epsilon: 0.0157 kilocalorie / mole  id: n13  rmin_half: 0.6 angstrom  >, <vdWType with smirks: [#6:1]  epsilon: 0.1033185743622 kilocalorie / mole  id: n14  rmin_half: 1.95815324792 angstrom  >, <vdWType with smirks: [#6X2:1]  epsilon: 0.2681357838595 kilocalorie / mole  id: n15  rmin_half: 1.906103098598 angstrom  >, <vdWType with smirks: [#6X4:1]  epsilon: 0.1205698919337 kilocalorie / mole  id: n16  rmin_half: 1.901434475347 angstrom  >, <vdWType with smirks: [#8:1]  epsilon: 0.2245605099459 kilocalorie / mole  id: n17  rmin_half: 1.701930728788 angstrom  >, <vdWType with smirks: [#8X2H0+0:1]  epsilon: 0.08532552033817 kilocalorie / mole  id: n18  rmin_half: 1.702425033604 angstrom  >, <vdWType with smirks: [#8X2H1+0:1]  epsilon: 0.1353608645661 kilocalorie / mole  id: n19  rmin_half: 1.697006198763 angstrom  >, <vdWType with smirks: [#7:1]  epsilon: 0.1025954704049 kilocalorie / mole  id: n20  rmin_half: 1.845362249921 angstrom  >, <vdWType with smirks: [#16:1]  epsilon: 0.25 kilocalorie / mole  id: n21  rmin_half: 2.0 angstrom  >, <vdWType with smirks: [#15:1]  epsilon: 0.2 kilocalorie / mole  id: n22  rmin_half: 2.1 angstrom  >, <vdWType with smirks: [#9:1]  epsilon: 0.061 kilocalorie / mole  id: n23  rmin_half: 1.75 angstrom  >, <vdWType with smirks: [#17:1]  epsilon: 0.2378672481785 kilocalorie / mole  id: n24  rmin_half: 1.847209758547 angstrom  >, <vdWType with smirks: [#35:1]  epsilon: 0.3359052482848 kilocalorie / mole  id: n25  rmin_half: 1.964485358405 angstrom  >, <vdWType with smirks: [#53:1]  epsilon: 0.4 kilocalorie / mole  id: n26  rmin_half: 2.35 angstrom  >, <vdWType with smirks: [#3+1:1]  epsilon: 0.0279896 kilocalorie / mole  id: n27  rmin_half: 1.025 angstrom  >, <vdWType with smirks: [#11+1:1]  epsilon: 0.0874393 kilocalorie / mole  id: n28  rmin_half: 1.369 angstrom  >, <vdWType with smirks: [#19+1:1]  epsilon: 0.1936829 kilocalorie / mole  id: n29  rmin_half: 1.705 angstrom  >, <vdWType with smirks: [#37+1:1]  epsilon: 0.3278219 kilocalorie / mole  id: n30  rmin_half: 1.813 angstrom  >, <vdWType with smirks: [#55+1:1]  epsilon: 0.4065394 kilocalorie / mole  id: n31  rmin_half: 1.976 angstrom  >, <vdWType with smirks: [#9X0-1:1]  epsilon: 0.003364 kilocalorie / mole  id: n32  rmin_half: 2.303 angstrom  >, <vdWType with smirks: [#17X0-1:1]  epsilon: 0.035591 kilocalorie / mole  id: n33  rmin_half: 2.513 angstrom  >, <vdWType with smirks: [#35X0-1:1]  epsilon: 0.0586554 kilocalorie / mole  id: n34  rmin_half: 2.608 angstrom  >, <vdWType with smirks: [#53X0-1:1]  epsilon: 0.0536816 kilocalorie / mole  id: n35  rmin_half: 2.86 angstrom  >, <vdWType with smirks: [#1]-[#8X2H2+0:1]-[#1]  epsilon: 0.1521 kilocalorie / mole  id: n-tip3p-O  sigma: 3.1507 angstrom  >, <vdWType with smirks: [#1:1]-[#8X2H2+0]-[#1]  epsilon: 0.0 kilocalorie / mole  id: n-tip3p-H  sigma: 1 angstrom  >, <vdWType with smirks: [#54:1]  epsilon: 0.561 kilocalorie / mole  id: n36  sigma: 4.363 angstrom  >]


From here you can inspect all the way down to individual parameters, which are stored in custom objects (in this case, `vdWType`). Let's look at the type with id `n16`, which looks like a generic carbon with four bonded neighbors:


```python
vdw_type = vdw_handler.parameters[15]
vdw_type
```




    <vdWType with smirks: [#6X4:1]  epsilon: 0.1205698919337 kilocalorie / mole  id: n16  rmin_half: 1.901434475347 angstrom  >



Note that the type contains both the physical parameters (sigma and epsilon, for a conventional 12-6 Lennard-Jones potential), but also an associated SMIRKS pattern. This particular SMIRKS pattern is fairly simple, but some can get much more complex.

The toolkit uses these SMIRKS patterns and direct chemical perception to assign parameters to particular atoms (or bonds, angles, etc.).

We'll use OpenFF 2.3.0 for the remainder of this tutorial. This is OpenFF's latest small molecule force field and is a leading open-source small molecule force field which [performs comparably to other open-source force fields](https://doi.org/10.1021/acs.jctc.3c00039). You can learn more about this and other SMIRNOFF force fields below:
<details>
  <summary><b>Click here to learn about available and planned SMIRNOFF force fields</b></summary>

# Existing force fields

## From OpenFF

### smirnoff99Frosst

This [force field](https://github.com/openforcefield/smirnoff99Frosst) is mostly a historical artifact today. It is the first SMIRNOFF force field, dating back to 2016. It is based on Merck-Frosst's [parm@frosst](http://www.ccl.net/cca/data/parm_at_Frosst/) and an old AMBER force field, parm99, which predates GAFF.

It is not recommended for general use today, but you might see it in papers that compare the performance of different force fields.

### Parsley

The Parsley line of force fields (`openff-1.y.z.offxml`) was OpenFF's [first full force field release](https://openforcefield.org/community/news/general/introducing-openforcefield-1.0/). Based on `smirnoff99Frosst`, these force fields are primarily re-fits of valence parameters using a large number of QM structures pulled from QCArchive. The first version was 1.0.0 and subsequent re-fits produced versions 1.1.0, 1.2.0, and 1.3.0. More detail is provided in an [associated paper](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00571).

### Sage

The Sage line of force fields (`openff-2.y.z.offxml`) continued the process of fitting to more (and more diverse) QM datasets, but also included a re-fit of the Lennard-Jones parameters. Small molecule geometries and energies [improved, in general,](https://openforcefield.org/community/news/general/sage2.0.0-release/) significantly over Parsley. These improvements notably transferred to protein-ligand binding free energies despite Sage not being specifically fit to them. For more, see the [associated paper](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00039).

[Subsequent releases](https://github.com/openforcefield/openff-forcefields/releases) used different fitting procedures and tweaks to parameter typing to improve performance and address issues with several specific chemistries. Notably, Sage 2.3.0 includes fast graph neural network charge assignment with AshGC, which is discussed later in this notebook. This charge model is trained to reproduce AM1-BCC charges, but scales 𝒪(N) rather than the 𝒪(N<sup>2</sup>) of common AM1-BCC implementations, making it suitable for large (>> 100 atoms) molecules. The latest release, **Sage 2.3.0 (`openff-2.3.0.offxml`) is the recommended force field for small molecule studies.**

## Ports

### Water models

OpenFF has ported [several existing water models](https://github.com/openforcefield/openff-forcefields/blob/main/docs/water-models.md) to SMIRNOFF format, including:

- TIP3P
- TIP3P-FB
- TIP4P-FB
- OPC
- OPC3

Existing main-line OpenFF force fields are fit against TIP3P water, so use of others is not (currently) recommended. This might change in the future, or OpenFF might even fit a new water model in a future release.

## ff14SB

OpenFF, in collaboration with Dave Cerutti of the Amber community, created a port of [ff14SB](https://pubs.acs.org/doi/10.1021/acs.jctc.5b00255), a popular Amber protein force field. There are some small numerical differences with how improper torsions are evaluated, but all other terms reproduce a canonical Amber source to high accuracy. **This is the only protein force field currently in SMIRNOFF (`.offxml`) format** and therefore the current recommendation for use with proteins. Primarily for technical reasons, porting other Amber force fields is not planned.

## Rosemary Alpha

A future line of force fields from OpenFF (code name "Rosemary", starting with `openff-3.0.0.offxml`) is intended to handle small molecules and biopolymers in a _self-consistent_ manner. This is exciting as it will streamline simulations of proteins with non-canonical amino acids! See the workshop [Simulating Post-Translationally Modified Proteins with the OpenFF Rosemary Alpha](https://github.com/openforcefield/2026-virtual-workshops/blob/main/ptm/ptm-workshop.ipynb). The first release will handle proteins, but future versions may cover nucleic acids. The performance, depending on the metrics used, is hoped to be comparable with existing Amber-family protein force fields. 

A pre-release version of Rosemary is available for testing as [`openff_no_water-3.0.0-alpha0.offxml`](https://github.com/openforcefield/openff-forcefields/blob/main/openforcefields/offxml/openff_no_water-3.0.0-alpha0.offxml). If you use it, see [the release notes](https://github.com/openforcefield/openff-forcefields/releases/tag/2025.10.1). 


## Non-main-line force fields

### SMIRNOFF plugins

https://github.com/openforcefield/smirnoff-plugins

https://github.com/jthorton/de-forcefields

# Future force fields

## From OpenFF

### Rosemary
There is no specific release date planned for the first full version of Rosemary, but it may be available in late 2026.

### Virtual sites

Another release from OpenFF may include some virtual site parameters with off-center charges. No release date is planned, but most of the supporting infrastructure is currently in place and some early studies have shown promise for better representing electrostatics of chemistries such as halogens and aromatic nitrogens.

## From you!

Anybody can write a SMIRNOFF force field! This workshop doesn't have time to cover force field _fitting_, but there are plenty of freely-available tools used today that can re-fit existing force fields or generate something new from the ground up. Once you've fit a new force field, a small Python package can distribute it in a way that the toolkit can [automatically load](https://docs.openforcefield.org/projects/toolkit/en/stable/faq.html#how-can-i-distribute-my-own-force-fields-in-smirnoff-format)!
</details>

<a id="topology"></a>
## 2. The `Topology` class represents a chemical system containing one or more `Molecule`s

Now we've loaded our desired force field (OpenFF 2.3.0), we need to specify the chemical system we want to assign force field parameters to ("parameterise"). Our system will be represented by a `Topology`, which we will build from one or more `Molecule`s. 

As a simple example, let's build a `Topology` containing an small molecule with some features which illustrate how parameters are applied according to SMIRKS matches. We'll use the crotonate anion, but you could draw any molecule you like and convert it to a SMILES string using tools like ChemDraw and [MolView](https://molview.org/).


```python
from openff.toolkit import Molecule, Topology

molecule = Molecule.from_smiles("C/C=C/C(=O)[O-]")
molecule
```


    
![svg](output_17_0.svg)
    


We can also visualise our molecule in 3D using NGLView, but only after generating 3D coordinates with `generate_conformers`:


```python
molecule.generate_conformers(n_conformers=1)
molecule.visualize(backend="nglview")
```


    NGLWidget()


<div class="alert alert-warning" style="max-width: 700px; margin-left: auto; margin-right: auto;">
    ⚠️ Be careful when creating molecules from SMILES with undefined stereochemistry. By default, an `UndefinedStereochemistryError` will be raised, but this can be downgraded to a warning by setting the <code>allow_undefined_stereo=True</code>. This will create a molecule with undefined stereochemistry, which might lead to incorrect parameterisation or surprising conformer generation. See the <a href="https://docs.openforcefield.org/en/latest/faq.html">"I'm getting stereochemistry errors when loading a molecule from a SMILES string" FAQ</a> for more details.
</div>

Topologies can always be assembled by constructing individual molecules and adding them together; these methods are for making common operations easier.

To convert a single `Molecule` to a `Topology`, you can use either `Molecule.to_topology()` or `Topology.from_molecules`


```python
topology = molecule.to_topology()

# Or, equivalently:
topology = Topology.from_molecules(molecules=[molecule])
```

From here we can add as many other molecules as we wish. For example, we can create a water molecule and add it to a topology 100 times.


```python
water = Molecule.from_smiles("O")
topology_with_water = Topology(topology)

for index in range(100):
    topology_with_water.add_molecule(water)

topology_with_water.n_molecules
```




    101



<div class="alert alert-info" style="max-width: 500px; margin-left: auto; margin-right: auto;">
    ℹ️ Positions are <i>optional</i> in <code>Molecule</code> (any by extension <code>Topology</code>) objects, so visualizing this topology in 3D doesn't make sense. Using it in a simulation would requiring assigning positions using a tool like Packmol or PDBFixer. Running simulations will be discussed later.
</div>

Keeping in mind that topologies are just collections of molecules, we can look up individual molecules by index in the `Topology.molecule()` function.


```python
topology_with_water.molecule(0), topology_with_water.molecule(1), topology_with_water.molecule(-1)
```




    (Molecule with name '' and SMILES '[H]/[C]([C](=[O])[O-])=[C](/[H])[C]([H])([H])[H]',
     Molecule with name '' and SMILES '[H][O][H]',
     Molecule with name '' and SMILES '[H][O][H]')



<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Build a <code>Topology</code> containing an MCL-1 ligand. Create the <code>Molecule</code> from an SDF file  (take a look at the docstring of <code>Molecule</code> to see how this can be done, noting that you can just pass the sdf path given below and don't need the <code>get_data_file_path</code> function). Also, see <a href="https://docs.openforcefield.org/projects/toolkit/en/stable/users/molecule_cookbook.html">Molecule cookbook</a> for all the ways to make a <code>Molecule</code>. The crystallographic MCL-1 ligand from PDB ID 6o6f is provided at <code>../structures/6o6f_ligand.sdf</code>. Note that you don't need to use <code>get_data_file_path</code> as we already know the path.
</div>



```python
# Inspect the Molecule docstring
mcl1_mol = Molecule("../structures/6o6f_ligand.sdf")
top = mcl1_mol.to_topology()
```

We will cover creating a topology for a protein-ligand complex in the next notebook.

<a id="interchange"></a>
## 3. `Interchange` objects contain fully parameterised systems with all the information needed to start a simulation

Now we've specified our force field and our chemical system using classes from the OpenFF Tools package (`ForceField`, `Molecule`, and `Topology`), and we want to apply our force field to our chemical topologies (parameterisation).

To do this, we'll use the `Interchange` class from the OpenFF Interchange package, which stores a fully-parameterised molecular system and provides methods to write out simulation-ready input files for a number of software packages. They key objective of Interchange is to provide an intermediate inspectable state after parameterisation and before conversion to an engine-specific format. For most users, an `Interchange` forms the bridge between the OpenFF ecosystem and their simulation software of choice. The current focus is applying SMIRNOFF force fields to chemical topologies and exporting the result to engines preferred by our users. In order of stability, OpenMM, GROMACS, Amber, and LAMMPS are supported. Future development may include support for CHARMM and other engines.

First, let's recreate our `molecule` and `topology` in case you overwrote them during the previous exercises:


```python
molecule = Molecule.from_smiles("C/C=C/C(=O)[O-]")
molecule.generate_conformers(n_conformers=1)
topology = molecule.to_topology()
```

An `Interchange` is most commonly constructed via the `Interchange.from_smirnoff()` class method. This method takes a SMIRNOFF force field and applies it to a molecular topology. 




```python
from openff.interchange import Interchange

Interchange.from_smirnoff?
```


```python
interchange = Interchange.from_smirnoff(
    force_field=sage,
    topology=topology,
)
interchange
```




    Interchange with 7 collections, non-periodic topology with 11 atoms.



<div class="alert alert-info" style="max-width: 700px; margin-left: auto; margin-right: auto;">
    ℹ️ <code>ForceField.create_interchange(topology)</code> and <code>Interchange.from_smirnoff(force_field, topology)</code> do the same thing - one just wraps the other. You can use whichever, and interpret them as substitutes of one another.
</div>

An `Interchange` object stores all information known about a system; this includes its chemistry, how that chemistry is represented by a force field, and how the system is organized in 3D space. It has five components:

1. **Topology**: Stores chemical information, such as connectivity and formal charges, independently of force field
1. **Collections**: Maps the chemical information to force field parameters. The force field itself is not directly stored
1. **Positions** (optional): Cartesian co-ordinates of atoms
1. **Box vectors** (optional): Periodicity information
1. **Velocities** (optional): Cartesian velocities of atoms

Let's inspect each of these.

The `Interchange.topology` attribute carries an object of the same type provided by the toolkit and therefore provides the same API. (In the future this may change).


```python
(
    interchange.topology.n_atoms,
    interchange.topology.n_bonds,
    interchange.topology.molecule(0).to_smiles(),
)
```




    (11, 10, '[H]/[C]([C](=[O])[O-])=[C](/[H])[C]([H])([H])[H]')



The `Interchange.collections` attribute carries a dictionary mapping handler names to `SMIRNOFFCollection` objects. These carry the physical parameters derived from applying the force field to the topology.


```python
[(key, value) for key, value in interchange.collections.items()]
```




    [('Bonds',
      Handler 'Bonds' with expression 'k/2*(r-length)**2', 10 mapping keys, and 6 potentials),
     ('Constraints',
      Handler 'Constraints' with expression '', 5 mapping keys, and 2 potentials),
     ('Angles',
      Handler 'Angles' with expression 'k/2*(theta-angle)**2', 15 mapping keys, and 5 potentials),
     ('ProperTorsions',
      Handler 'ProperTorsions' with expression 'k*(1+cos(periodicity*theta-phase))', 19 mapping keys, and 7 potentials),
     ('ImproperTorsions',
      Handler 'ImproperTorsions' with expression 'k*(1+cos(periodicity*theta-phase))', 9 mapping keys, and 2 potentials),
     ('vdW',
      Handler 'vdW' with expression '4*epsilon*((sigma/r)**12-(sigma/r)**6)', 11 mapping keys, and 5 potentials),
     ('Electrostatics',
      Handler 'Electrostatics' with expression 'coul', 11 mapping keys, and 11 potentials)]



Note that each `SMIRNOFFCollection` specifies an algebraic expression which is used to compute the potential energy.

Let's quickly visualize this molecule with atom indices -- this is helpful for looking up particular parameters.


```python
from rdkit.Chem import Draw
from openff.toolkit.topology import Molecule
from IPython.display import SVG


def mol_with_atom_index(molecule: Molecule, width: int = 300, height: int = 300) -> str:
    molecule_copy = Molecule(molecule)
    molecule_copy._conformers = None

    rdmol = molecule_copy.to_rdkit()

    # Build labels like "C:0", "C:1", "C:2", ...
    atom_labels = {
        atom.GetIdx(): f"{atom.GetSymbol()}:{atom.GetIdx()}"
        for atom in rdmol.GetAtoms()
    }

    drawer = Draw.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    for idx, label in atom_labels.items():
        opts.atomLabels[idx] = label

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, rdmol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


SVG(mol_with_atom_index(molecule))
```




    
![svg](output_44_0.svg)
    



The `key_map` attribute of a `SMIRNOFFCollection` maps a `TopologyKey` (such as a `BondKey`) to a `PotentialKey`, which identifies unique parameters:


```python
collection = interchange.collections["Bonds"]
collection.key_map
```




    {BondKey with atom indices (0, 1): PotentialKey associated with handler 'Bonds' with id '[#6X4:1]-[#6X3:2]',
     BondKey with atom indices (0, 6): PotentialKey associated with handler 'Bonds' with id '[#6X4:1]-[#1:2]',
     BondKey with atom indices (0, 7): PotentialKey associated with handler 'Bonds' with id '[#6X4:1]-[#1:2]',
     BondKey with atom indices (0, 8): PotentialKey associated with handler 'Bonds' with id '[#6X4:1]-[#1:2]',
     BondKey with atom indices (1, 2): PotentialKey associated with handler 'Bonds' with id '[#6X3:1]=[#6X3:2]',
     BondKey with atom indices (1, 9): PotentialKey associated with handler 'Bonds' with id '[#6X3:1]-[#1:2]',
     BondKey with atom indices (2, 3): PotentialKey associated with handler 'Bonds' with id '[#6X3:1]-[#6X3:2]',
     BondKey with atom indices (2, 10): PotentialKey associated with handler 'Bonds' with id '[#6X3:1]-[#1:2]',
     BondKey with atom indices (3, 4): PotentialKey associated with handler 'Bonds' with id '[#6X3:1](~[#8X1])~[#8X1:2]',
     BondKey with atom indices (3, 5): PotentialKey associated with handler 'Bonds' with id '[#6X3:1](~[#8X1])~[#8X1:2]'}



We can see that the C=C bond (indices (1,2)) is associated with a potential key with the SMIRKS pattern `[#6X3:1]=[#6X3:2]` (specifying any two carbons each bonded to 3 atoms and connected by a double bond). Note that the (1,0) C-C bond is matched by the SMRIKS `[#6X3:1]-[#6X3:2]`, which specifies the atoms in the same way, showing that the parameters have been assigned by directly using information about the bond. This contrasts to traditional atom typing approaches, where information about the bond would be implicitly encoded in the atom types used to assign the parameters. Another example of this "direct chemical perception" is the assignment of the carboxylate carbon-oxygen bond parameters, which only match (triply-connected carbon) - (singly-connnected oxygen) bonds when the carbon is bonded to another singly-connected oxygen.

To see the actual parmeters specified for this bond, we can look up the `Potential` objects using the `PotentialKey`s.


```python
for topology_key, potential_key in collection.key_map.items():
    potential = collection.potentials[potential_key]
    print(f"{topology_key} -> {potential}")
```

    atom_indices=(0, 1) bond_order=None -> parameters={'k': <Quantity(590.299542, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.50405326, 'angstrom')>} map_key=None
    atom_indices=(0, 6) bond_order=None -> parameters={'k': <Quantity(680.766445, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.09244581, 'angstrom')>} map_key=None
    atom_indices=(0, 7) bond_order=None -> parameters={'k': <Quantity(680.766445, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.09244581, 'angstrom')>} map_key=None
    atom_indices=(0, 8) bond_order=None -> parameters={'k': <Quantity(680.766445, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.09244581, 'angstrom')>} map_key=None
    atom_indices=(1, 2) bond_order=None -> parameters={'k': <Quantity(911.750546, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.36599361, 'angstrom')>} map_key=None
    atom_indices=(1, 9) bond_order=None -> parameters={'k': <Quantity(799.445393, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.08663579, 'angstrom')>} map_key=None
    atom_indices=(2, 3) bond_order=None -> parameters={'k': <Quantity(535.332596, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.46005396, 'angstrom')>} map_key=None
    atom_indices=(2, 10) bond_order=None -> parameters={'k': <Quantity(799.445393, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.08663579, 'angstrom')>} map_key=None
    atom_indices=(3, 4) bond_order=None -> parameters={'k': <Quantity(1141.6952, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.25831056, 'angstrom')>} map_key=None
    atom_indices=(3, 5) bond_order=None -> parameters={'k': <Quantity(1141.6952, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.25831056, 'angstrom')>} map_key=None


So our C=C bond (indices (1,2)) has a force constant of 904 kcal mol<sup>-1</sup> Å<sup>-2</sup> and an equilibrium bond length of 1.37 Å. Note that the [`ForceField.label_molecules`](https://docs.openforcefield.org/projects/toolkit/en/stable/api/generated/openff.toolkit.typing.engines.smirnoff.ForceField.html#openff.toolkit.typing.engines.smirnoff.ForceField.label_molecules) method is also useful for checking which parameters will be applied to your molecule.

<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Have a look at the "Angles", "ProperTorsions", and "ImproperTorsions" applied. Where are the "ImproperTorsions" applied and why?
</div>


```python
# For example, the ImproperTorsions collection:
collection = interchange.collections["ImproperTorsions"]
collection.key_map

# These enforce planarity around the double bond and carboxylate group
```




    {ImproperTorsionKey with atom indices (1, 0, 2, 9), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (1, 2, 9, 0), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (1, 9, 0, 2), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (2, 1, 3, 10), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (2, 3, 10, 1), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (2, 10, 1, 3), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (3, 2, 4, 5), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]', mult 0,
     ImproperTorsionKey with atom indices (3, 4, 5, 2), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]', mult 0,
     ImproperTorsionKey with atom indices (3, 5, 2, 4), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]', mult 0}



<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Change the molecule from the anion to the neutral form by adding a hydrogen to the SMILES. How does this affect the bond strengths of the two carboxyl oxygens?
</div>


```python
bond1_indices, bond2_indices = (3,4), (3,5)
smiles = {"anion": "C/C=C/C(=O)[O-]", "neutral": "C/C=C/C(=O)O"}

# Use fresh variable names so we don't overwrite `molecule`, `topology` and `interchange`
for name, smiles in smiles.items():
    print(f"\n{name} ({smiles}):")
    protonation_state = Molecule.from_smiles(smiles)
    protonation_state_interchange = Interchange.from_smirnoff(
        force_field=sage,
        topology=protonation_state.to_topology(),
    )
    bonds = protonation_state_interchange.collections["Bonds"]
    bond1 = bonds[bond1_indices]
    bond2 = bonds[bond2_indices]
    print(f"  Bond {bond1_indices}: {bond1}")
    print(f"  Bond {bond2_indices}: {bond2}")
```

    
    anion (C/C=C/C(=O)[O-]):
      Bond (3, 4): parameters={'k': <Quantity(1141.6952, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.25831056, 'angstrom')>} map_key=None
      Bond (3, 5): parameters={'k': <Quantity(1141.6952, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.25831056, 'angstrom')>} map_key=None
    
    neutral (C/C=C/C(=O)O):


      Bond (3, 4): parameters={'k': <Quantity(1635.44354, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.23024478, 'angstrom')>} map_key=None
      Bond (3, 5): parameters={'k': <Quantity(676.222227, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.35758125, 'angstrom')>} map_key=None


Finally, `interchange.box` and `interchange.velocities` are `None`, although `interchange.positions` is populated because we passed a topology with a molecule that had a defined conformer, so `from_smirnoff` set atomic positions from this information:


```python
interchange.positions, interchange.box, interchange.velocities
```




    (<Quantity([[-0.16884651 -0.03204099  0.02797506]
      [-0.02695262 -0.04551754 -0.01496804]
      [ 0.05777792  0.05074697  0.01721889]
      [ 0.19690544  0.03719656 -0.02507113]
      [ 0.23595445 -0.06303954 -0.09039396]
      [ 0.29175033  0.13419756  0.00498716]
      [-0.1829386   0.05336654  0.09439656]
      [-0.23077479 -0.01166554 -0.06346681]
      [-0.20531354 -0.1285518   0.07293682]
      [ 0.00522258 -0.13138426 -0.0706145 ]
      [ 0.02721534  0.13669204  0.07249894]], 'nanometer')>,
     None,
     None)



An `Interchange` handles all the information required to run a simulation and allows us to export input files for our engine of choice (OpenMM, GROMACS, LAMMPS, and Amber are all supported)! Let's run a simulation.

Note that since the `Interchange` only contains crontonate and has no box vectors, this would correspond to a vacuum simulation.  We can use [`PACKMOL`](https://m3g.github.io/packmol/) to generate initial positions for a box of water and our solute. Let's use neutral crotonoic acid as our solute so we don't have to worry about neutralising the box.


```python
from openff.interchange.components._packmol import UNIT_CUBE, pack_box
from openff.toolkit import unit

water = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
solute = Molecule.from_smiles("C/C=C/C(=O)[OH]") # neutral crotonoic acid

# Naming the residue is not needed to parameterize the system or run the simulation, but makes visualization easier
for atom in water.atoms:
    atom.metadata["residue_name"] = "HOH"

# Generate initial positions using OpenFF's PACKMOL interface. Note that
# using a cubic box is a simple but inefficient choice -- a rhombic
# dodecahedron that provides the same solute separation has only ~ 71 % of
# the volume.
topology = pack_box(
    molecules=[solute, water],
    number_of_copies=[1, 1400],
    box_vectors=3.5 * UNIT_CUBE * unit.nanometer,
)

# Parameterise with Sage, which contains parameters for TIP3P water
interchange = Interchange.from_smirnoff(force_field=sage, topology=topology)
interchange.topology.n_atoms, interchange.box, interchange.positions.shape
```




    (4212,
     <Quantity([[3.5 0.  0. ]
      [0.  3.5 0. ]
      [0.  0.  3.5]], 'nanometer')>,
     (4212, 3))



At this point, we could easily export input files for our simulation engine of choice. For example, for Amber:


```python
interchange.to_amber(prefix="ligand")
```

    /opt/conda/envs/openff-env/lib/python3.14/site-packages/openff/interchange/components/mdconfig.py:504: UserWarning: Ambiguous failure while processing constraints. Constraining h-bonds as a stopgap.
      warnings.warn(
    /opt/conda/envs/openff-env/lib/python3.14/site-packages/openff/interchange/components/mdconfig.py:434: SwitchingFunctionNotImplementedWarning: A switching distance 8.0 angstrom was specified by the force field, but Amber does not implement a switching function. Using a hard cut-off instead. Non-bonded interactions will be affected.
      warnings.warn(



```python
# Check the new files
! ls
```

    complex.gro		 protein_ligand_complex_parameterisation_and_md.ipynb
    complex.top		 small_molecule_parameterisation.ipynb
    complex_pointenergy.mdp  topology.json
    ligand.inpcrd		 trajectory.dcd
    ligand.prmtop		 trajectory_gpu.dcd
    ligand_pointenergy.in	 trajectory_showcase.dcd


Here, we'll export to OpenMM and run a short simulation directly from the noteboook. We can create an OpenMM `Simulation` object from the `Interchange` and run for a specified wall clock time using `runForClockTime` (the simluation time will depend on how quickly it runs on your machine). We keep the volume ($V$), number of particles ($N$), and average temperature ($T$) (using the LangevinIntegrator) constant and the simulation corresponds to the $NVT$ ensemble.


```python
import openmm
import openmm.app
import openmm.unit
from openff.interchange import Interchange
import MDAnalysis as mda
import nglview


def run_openmm(
    interchange: Interchange,
    reporter_frequency: int = 50, # Decrease this to save more frames!
    trajectory_name: str = "small_mol_solvated.dcd",
):
    simulation = interchange.to_openmm_simulation(
        integrator=openmm.LangevinIntegrator(
            300 * openmm.unit.kelvin,
            1 / openmm.unit.picosecond,
            0.002 * openmm.unit.picoseconds,
        ),
    )

    dcd_reporter = openmm.app.DCDReporter(trajectory_name, reporter_frequency)
    simulation.reporters.append(dcd_reporter)

    simulation.context.setVelocitiesToTemperature(300 * openmm.unit.kelvin)
    simulation.runForClockTime(10 * openmm.unit.second)


def visualise_traj(
    topology: Topology, filename: str = "small_mol_solvated.dcd"
) -> nglview.NGLWidget:
    """Visualise a trajectory using nglview."""

    u = mda.Universe(topology.to_openmm(), filename)

    view = nglview.show_mdanalysis(u)
    view.add_representation("licorice", selection="water")

    return view


run_openmm(interchange)
visualise_traj(interchange.topology)
```

    /opt/conda/envs/openff-env/lib/python3.14/site-packages/MDAnalysis/coordinates/DCD.py:171: DeprecationWarning: DCDReader currently makes independent timesteps by copying self.ts while other readers update self.ts inplace. This behavior will be changed in 3.0 to be the same as other readers. Read more at https://github.com/MDAnalysis/mdanalysis/issues/3889 to learn if this change in behavior might affect you.
      warnings.warn("DCDReader currently makes independent timesteps"



    NGLWidget(max_frame=25)


<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> (Only complete this if you have time -- otherwise proceed to section 4.) Create an Interchange object for an MCL-1 ligand. Inspect the parameters assigned and run short simulation as above.
</div>

<a id="gnn_charges"></a>
## 4. Graph Neural Networks Allow Fast Assignment of Partial Charges

You might notice that Sage force fields don't contain tabulated charges for most atomic environments in the way they do for all other terms in the force field. For example, [Sage 2.2.1](https://github.com/openforcefield/openff-forcefields/blob/main/openforcefields/offxml/openff-2.2.1.offxml) instead specifies:
```
<ToolkitAM1BCC version="0.3"></ToolkitAM1BCC>
```
which means that partial charges will be calculated using the common AM1-BCC method. Charges from a semi-empirical quantum chemistry calculation (Austin Model 1) are corrected (bond charge correction) to approximate charges obtained by fitting to the electrostatic potential at the HF/6-31G* level (see [Jakalian et al.](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1096-987X(20000130)21:2%3C132::AID-JCC5%3E3.0.CO;2-P)). Unfortunately, parameterisation with AM1-BCC using OpenEye or AmberTools scales 𝒪(N<sup>2</sup>) in the number of atoms N, making it prohibitively slow for large molecules and biopolymers.

Methods which assign partial charges using graph neural networks offer rapid assignment with better scaling. They also offer the possibility of going beyond traditionally affordable QM levels of theory by training to quickly reproduce charges from expensive calculations. For example, OpenFF's [AshGC](https://doi.org/10.1021/acs.jctc.6c00169) model is fit to AM1-BCC charges and offers 𝒪(N) scaling, while [Adams et al.](https://doi.org/10.1021/acs.jctc.5c01520) trained models to reproduce atoms-in-molecules charges and electrostatic potentials obtained at a high level of theory. AshGC is used by [Sage 2.3.0](https://github.com/openforcefield/openff-forcefields/blob/main/openforcefields/offxml/openff-2.3.0.offxml) -- if you inspect the file, you'll see:
```
<NAGLCharges model_file="openff-gnn-am1bcc-1.0.0.pt" model_file_hash="7981e7f5b0b1e424c9e10a40d9e7606d96dcd3dd2b095cb4eeff6829f92238ee" version="0.3"></NAGLCharges>
```
where "openff-gnn-am1bcc-1.0.0.pt" is the AshGC model.

The GNN charge model is the main difference between Sage 2.2.1 and 2.3.0. Here, we'll compare the parameterisation speed and charges obtained using each force field.


```python
from openff.toolkit import Molecule

molecule = Molecule("../structures/6o6f_ligand.sdf")
```

First, let's parameterise with Sage 2.2.1, which uses the traditional AM1-BCC model, and check how long this takes...


```python
%%time
sage221 = ForceField("openff-2.2.1.offxml")
interchange_sage221 = Interchange.from_smirnoff(force_field=sage221, topology=molecule.to_topology())
```

    CPU times: user 245 ms, sys: 6.57 ms, total: 252 ms
    Wall time: 14.9 s


Note that repeating these cells will show much faster assignment as partial charges are cached for a given molecule and charge method.

Now, let's try Sage 2.3.0, which uses AshGC charges:


```python
%%time
sage230 = ForceField("openff-2.3.0.offxml")
interchange_sage230 = Interchange.from_smirnoff(force_field=sage230, topology=molecule.to_topology())
```

    CPU times: user 1.29 s, sys: 20.6 ms, total: 1.31 s
    Wall time: 1.25 s


<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Compare the charges obtained with AM1-BCC and AshGC by inspecting the electrostatics collection of each <code>Interchange</code> object (see Section 3). How big are these differences on average? What is the largest difference? Which atom are these on? The <code>np.max</code> function may be useful.
</div>


```python
# Compare charges assigned with AM1-BCC (Sage 2.2.1) and AshGC (Sage 2.3.0)...
import numpy as np

charges_am1bcc = np.array([c.m for c in interchange_sage221["Electrostatics"].charges.values()])
charges_ashgc = np.array([c.m for c in interchange_sage230["Electrostatics"].charges.values()])

differences = charges_am1bcc - charges_ashgc
print(f"Mean absolute difference: {np.mean(np.abs(differences)):.4f} e")

max_index = int(np.argmax(np.abs(differences)))
print(
    f"Largest absolute difference is {np.abs(differences[max_index]):.4f} e, "
    f"for atom index {max_index}, which is a {molecule.atoms[max_index].symbol} atom"
)
```

    Mean absolute difference: 0.0131 e
    Largest absolute difference is 0.0560 e, for atom index 17, which is a C atom


<a id="summary"></a>
## 5. Conclusions

* The `ForceField` class from the OpenFF Toolkit allows force fields to be easily loaded and inspected.
* The `Molecule` and `Topology` classes from the OpenFF Toolkit allow us to represent a chemical system, independently from the force field.
* The `Interchange` class from OpenFF Interchange handles fully parameterised systems with all the information required to start a simulation. Exporting to the simluation engine of your choice is simple, and we can easily run a simulation with OpenMM without leaving the notebook!
* Graph neural networks can provide fast and high-quality conformer-independent partial charges.

<a id="further_materials"></a>
## 6. There's Lots More to OpenFF!

A variety of example notebooks for OpenFF software are provided [here](https://docs.openforcefield.org/en/latest/examples.html). A few which are particularly relevant are:

- [Compute conformer energies for a small molecule](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-toolkit/conformer_energies/conformer_energies.html)
- [Modifying a SMIRNOFF force field](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-toolkit/forcefield_modification/forcefield_modification.html)
- [Inspect parameters assigned to specific molecules](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-toolkit/inspect_assigned_parameters/inspect_assigned_parameters.html)

<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Extra Exercises:</b> Based on the above tutorials, can you:
            <ul>
            <li>Generate several conformers for one of your MCL-1 ligands and compute their relative energies using OpenFF 2.3.0?</li>
            <li>Modify OpenFF 2.3.0 to change some of the parameters applied to one of your MCL-1 ligands? Minimise the ligand with this new force field and see how your changes influence the conformation.</li>
            <li>Analyse which parameters are shared and which are only applied to one or few molecules for a set of MCL-1 ligands?</li>
            </ul>
</div>
