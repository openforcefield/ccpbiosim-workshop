# Parameterisation, Molecular Dynamics, and Trajectory Analysis of a Protein-Ligand Complex with OpenFF, OpenMM, MDAnalysis, and ProLIF

This is the second of two jupyter notebooks on handling force fields using [Open Force Field's](https://openforcefield.org/) software, and subsequent molecular dynamics and analysis. The first notebook (`small_molecule_parameterisation.ipynb`) introduced fundamental concepts in OpenFF and demonstrated parameterisation of a small molecule.
This notebook demonstrates how to prepare a system that combines solvent, a ligand using Sage, and a protein using a standard AMBER force field. We'll take the structures of the MCL-1 and the bound ligand from the crystal structure, but we could just as easily use a ligand pose from docking. We'll solvate the complex, assemble the system, parameterise it, and finally simulate it with OpenMM and visualize the results, all without leaving the notebook. Have fun!

### Prerequisites

 - Basic knowledge of Python
 - Basic familiarity with molecular mechanics force fields and molecular dynamics simulations (see talk by Danny Cole)
 - Completion of `small_molecule_parameterisation.ipynb`

### The Plan

| Action | Software|
|--|--|
| [Visualise the protein and ligand](#visualise) | NGLView |
| [Assemble the topology](#assemble) | OpenFF Toolkit
| [Parameterise the complex](#parameterise) | OpenFF Toolkit and OpenFF Interchange
| [Simulate the complex](#simulate) | OpenMM
| [Analyse pose stability and interactions](#analyse) | MDAnalysis and ProLIF
| [Review what you've learnt](#summary) | 
| [Check out other OpenFF tutorials](#further_materials) | 
| [Check out related non-OpenFF software](#further_non_openff_materials) | 

### Jupyter Cheat Sheet

- To run the currently highlighted cell and move focus to the next cell, hold <kbd>&#x21E7; Shift</kbd> and press <kbd>&#x23ce; Enter</kbd>;
- To run the currently highlighted cell and keep focus in the same cell, hold <kbd>&#x21E7; Ctrl</kbd> and press <kbd>&#x23ce; Enter</kbd>;
- To get help for a specific function, place the cursor within the function's brackets, hold <kbd>&#x21E7; Shift</kbd>, and press <kbd>&#x21E5; Tab</kbd>;

### Acknowledgements

Most of this material was adapted from:


* The OpenFF [toolkit showcase](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-toolkit/toolkit_showcase/toolkit_showcase.html)
* The [ProLIF Ligand-protein MD tutorial](https://prolif.readthedocs.io/en/latest/notebooks/md-ligand-protein.html#ligand-protein-md)

### Maintainers
 - Finlay Clark -- finlay.clark@newcastle.ac.uk (@fjclark)


<a id="visualise"></a>
## 1. We Can Visualise the Protein and Ligand with NGLView

We'll be using the MCL-1 complex with PDBID `6o6f`. MCL-1 is a common system for benchmarking protein-ligand binding free energy calcalulations and features in the [protein-ligand benchmark set hosted by OpenFF](https://github.com/openforcefield/protein-ligand-benchmark).

As we've already covered structure preparation, we provide pre-prepared protein and ligand structures in the `structures` directory. These are ready for simulation:

- Their co-ordinates are super-imposable (and there are no clashes between waters and the ligand)
- Hydrogens have been added to protein and crystallographic waters consistent with pH 7
- The protein's termini have been capped where appropriate to prevent unphysical charges
- A missing residue in the middle of the chain has been added
- The protein has been solvated and 150 mM NaCl added
- The overall system is neutral

If you'd like more information on how this was done, check out `structures/README.md`.


```python
receptor_path = "../structures/6o6f_protein_solvated.pdb"
ligand_path = "../structures/6o6f_ligand.sdf"
```

We can visualize each structure using the [NGLView] widget. These visualizations are interactive; rotate by dragging the left mouse button, pan with the right mouse button, and zoom with the scroll wheel. You can also mouse over an atom to see its details, and click an atom to center the view on it. When you mouse over the widget, a full screen button will appear in its top right corner.

[NGLView]: https://github.com/nglviewer/nglview


```python
import nglview

view = nglview.show_structure_file(ligand_path)
view
```


    



    NGLWidget()


<div class="alert alert-info" style="max-width: 700px; margin-left: auto; margin-right: auto;">
    ℹ️ Try replacing <code>ligand_path</code> with <code>receptor_path</code> to visualize the protein!
</div>


<a id="assemble"></a>
## 2. OpenFF Toolkit Allows Us to Assemble the Topology

Conceptually, this step involves putting together the positions of all of the components of the system. We'll create  a [`Topology`] to keep track of the contents of our system. As discussed in the previous notebook, `Topology` represents a collection of molecules; it doesn't have any association with any force field parameters.

[`Topology`]: https://docs.openforcefield.org/projects/toolkit/en/stable/api/generated/openff.toolkit.topology.Topology.html

First, we'll load the ligand and receptor into OpenFF Toolkit [`Molecule`] objects, which keep track of all their chemical information. As discussed previously, `Molecule` represents a collection of atoms with specified formal charges, connected by bonds with specified bond orders, optionally including any number of conformer coordinates. This is intended to closely align with a chemist's intuitive understanding of a molecule, rather than simply wrap the minimal information needed for a calculation.

SDF files include all a molecule's bond orders and formal charges, as well as coordinates, so they're ideal as a format for distributing small molecules. And that's exactly the format the ligand is stored in!

[`Molecule`]: https://docs.openforcefield.org/projects/toolkit/en/stable/api/generated/openff.toolkit.topology.Molecule.html


```python
from openff.toolkit import Molecule

# Load a molecule from a SDF file
ligand = Molecule.from_file(ligand_path)

# Print out a SMILES code for the ligand
print(ligand.to_smiles(explicit_hydrogens=False))

# Visualize the molecule
ligand.visualize(show_all_hydrogens=False)
```

    O=C([O-])c1ccc2c(c1)N(CC1CCC1)C[C@@]1(CCc3cc(Cl)ccc31)CO2





    
![svg](output_8_1.svg)
    



Conventionally, SDF files are used for ligands and PDB files are used for proteins. The toolkit loads polymers (including biopolymers such as proteins) via PDB files by inferring chemical information from the file and a known dictionary of common residues (and water, and ions). To do this, we'll use `Topology.from_pdb`


```python
from openff.toolkit import Topology

topology = Topology.from_pdb(receptor_path)

# Note that we have box vectors:
print(topology.box_vectors)
```

    [[6.5199 0.0 0.0] [0.0 6.5199 0.0] [0.0 0.0 6.5199]] nanometer


We can add the ligand `Molecule` to the topology created from the protein PDB file.


```python
topology.add_molecule(ligand)
```




    7923



Now that we've assembled our topology, we can save it to disk. We can use JSON for this, which makes it human readable in a pinch. This stores everything we've just assembled - molecular identities, conformers, box vectors, and everything else. The topology can then be loaded later on with the [`Topology.from_json()`] method. This is great for running the same system through different force fields, distribution with a paper, or for assembling systems in stages.

[`Topology.from_json()`]: https://docs.openforcefield.org/projects/toolkit/en/stable/api/generated/openff.toolkit.topology.Topology.html#openff.toolkit.topology.Topology.from_json


```python
with open("topology.json", "w") as f:
    print(topology.to_json(), file=f)
```

To visualize inside the notebook, we'll use `Topology.visualize`, which uses NGLview under the hood. NGLview supports a wide variety of [molecular visualization methods], as well as a VMD-like [atom selection language]. This can be used to visualize complex systems like this one.

The widget consists of a minimally documented [Python library frontend] and an extensively documented [JavaScript backend]. You'll need to refer to the documentation for both to do anything sophisticated, as the Python code delegates most of its options and functionality to the JS code.

By default, the toolkit attemps to draw some components with special representations:
* Waters: [line](https://nglviewer.org/ngl/api/manual/molecular-representations.html#line)
* Ions: [spacefill](https://nglviewer.org/ngl/api/manual/molecular-representations.html#spacefill)
* Proteins: [cartoon](https://nglviewer.org/ngl/api/manual/molecular-representations.html#cartoon)

Everything else (i.e. unrecognized ligands) are drawn with the [licorice](https://nglviewer.org/ngl/api/manual/molecular-representations.html#licorice) representation, which is basically a ball+stick model. A box representing the periodic boundary conditions is also added.

[molecular visualization methods]: https://nglviewer.org/ngl/api/manual/molecular-representations.html
[atom selection language]: https://nglviewer.org/ngl/api/manual/selection-language.html
[Python library frontend]: https://nglviewer.org/nglview/latest/api.html
[JavaScript backend]: https://nglviewer.org/ngl/api/manual/index.html


```python
view = topology.visualize()

# can make further modifications to this representation object, or just look at it
view
```


    NGLWidget()


<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
✏️ <b>Exercise:</b>️ Have a play with this visualization! Try clearing the default representations with <code>view.clear()</code> and configuring your own cartoon <em>(Hint: <a href=https://nglviewer.org/nglview/latest/api.html#nglview.NGLWidget>Check the docs</a>)</em>. You'll need <code>view.add_representation</code>. See if you can display the ligand in a way you like. When you're happy with what you've made, save the image with <code>view.download_image()</code>
</div>


```python
view = topology.visualize()

view.clear()
# Show the protein as a surface
view.add_representation("surface", selection="polymer", opacity=0.3)
# # Show the ligand as sticks
view.add_representation("licorice", selection="ligand")
view
```


    NGLWidget()


<a id="parameterise"></a>
## 3. We Can Assemble a Combined `ForceField` and use this to Parameterise the Whole System

Now that we've prepared our coordinates, we should choose the force field. For now, we don't have any single SMIRNOFF force field that can handle both proteins and small molecules. The Rosemary line of force fields (starting with `openff-3.0.0.offxml`) is intended to do exactly this. A pre-release version, [`openff_no_water-3.0.0-alpha0.offxml`](https://github.com/openforcefield/openff-forcefields/blob/main/openforcefields/offxml/openff_no_water-3.0.0-alpha0.offxml), is already available for testing (if you use it, see [the release notes](https://github.com/openforcefield/openff-forcefields/releases/tag/2025.10.1)). There is no specific release date planned for the first full version, but it may be available later in 2026. As an alternative, we'll combine the AMBER-compatible [Sage] small molecule force field with the SMIRNOFF port of AMBER ff14SB. Note that Sage also includes the TIP3P water model, which is appropriate for AMBER ff14SB too.

When we combine multiple SMIRNOFF force fields into one, we provide them in an order from general to specific. Sage includes parameters that could be applied to a protein, but they're general across all molecules; ff14SB's parameters are specific to proteins. Since the Toolkit always applies the last parameters that match a moiety, this order makes sure the right parameters get assigned.

[Sage]: https://openforcefield.org/force-fields/force-fields/#sage

<div class="alert alert-warning" style="max-width: 700px; margin-left: auto; margin-right: auto;">
⚠️ Warning: If your small molecule has an amino acid substructure in it, the specific patterns in the ff14SB force field will override the general ones from openff-2.3.0.offxml. This is the SMIRNOFF format being applied correctly, but some users may find this surprising, especially since terminal caps like ACE and NME are relatively small substructures and will sometimes appear in ligands.
</div>



```python
from openff.toolkit import ForceField

# Assemble the combined force field
sage_ff14sb = ForceField("openff-2.3.0.offxml", "ff14sb_off_impropers_0.0.3.offxml")
```

We now have a `Topology`, which stores the chemical information of the system, and a `ForceField`, which maps chemistry to force field parameters. To parametrize the system, we combine these two objects into an [`Interchange`], as discussed in the previous notebook.

An `Interchange` represents a completely parameterised molecular mechanics system. Partial charges are computed here according to the instructions in the force field: Sage 2.3.0 assigns the ligand's charges with the AshGC graph neural network model (see section 4 of the previous notebook), while ff14SB supplies library charges for the protein, water, and ions. This is also where virtual sites required by the force field will be introduced. This all happens behind the scenes; all we have to do is combine an abstract chemical description with a force field. This makes it easy to change water model or force field, as the chemistry being modelled is completely independent of the model itself.

[`Interchange`]: https://docs.openforcefield.org/projects/interchange/en/stable/_autosummary/openff.interchange.components.interchange.Interchange.html


```python
interchange = sage_ff14sb.create_interchange(topology)
```

*(This should take well under a minute, most of it spent on the chemical perception required by the AMBER protein force field port. It used to be considerably slower: assigning the ligand's AM1-BCC charges was the bottleneck, and AshGC has removed it.)*

Before we simulate, let's recap. We've constructed a `Topology` out of a number of `Molecule` objects, each of which represents a particular chemical independent of any model details. The `Topology` then represents an entire chemical system, which in theory could be modelled in any number of ways. Our `Topology` also includes atom positions and box vectors, but if we thought that was too concrete for our use case we could leave them out and add them after parameterisation.

Separately, we've constructed a `ForceField` by combining a general SMIRNOFF force field with a protein-specific SMIRNOFF force field. A SMIRNOFF force field is a bunch of rules for applying force field parameters to chemicals via SMARTS patterns. The force field includes everything needed to compute an energy: parameters, charges, functional forms, non-bonded methods and cutoffs, virtual sites, and so on.

Then, we've parameterised our `Topology` with our `ForceField` to produce an `Interchange`. This applies all our rules and gives us a system ready to simulate. An `Interchange` can also concretely define positions, velocities, and box vectors, whether they come from the `Topology` or are added after parameterisation. Once we have the `Interchange`, we can produce input data for any of the supported MM engines.

This clear delineation makes benchmarking the same system against different force fields or the same force field against different force fields easy. The SMIRNOFF format makes distributing force fields in an engine agnostic way possible. Everything is an open standard or written in open source Python, so we can see how it works and even change it if we need to.

<a id="simulate"></a>
## 4 We can Simulate the Dynamics of the Complex with OpenMM

To use an `Interchange`, we need to convert it to the input expected by a particular molecular mechanics engine. We'll use OpenMM, because its support is the most mature and the fastest, but GROMACS, LAMMPS, and Amber all also supported.

For example, we can write out GROMACS `complex.gro` and `complex.top` files with:


```python
interchange.to_gromacs(prefix="complex")
```

    /opt/conda/envs/openff-env/lib/python3.14/site-packages/openff/interchange/components/mdconfig.py:504: UserWarning: Ambiguous failure while processing constraints. Constraining h-bonds as a stopgap.
      warnings.warn(



```python
# Check the new gromacs output
!ls
```

    complex.gro
    complex.top
    complex_pointenergy.mdp
    protein_ligand_complex_parameterisation_and_md.ipynb
    small_molecule_parameterisation.ipynb
    topology.json
    trajectory_gpu.dcd



All that remains is to tell OpenMM the details about how we want to integrate and record data for the simulation, and then to put everything together and run it! The steps are:

1. Configure and run the simulation
1. Minimise the combined system
1. Run a short simulation
1. Visualise the trajectory

[preliminary support]: https://docs.openforcefield.org/projects/interchange/en/stable/using/output.html

### 4.1 Configure and run the simulation

Here, we'll use a Langevin thermostat at 300 Kelvin and a 2 fs time step. We'll write the structure to disk every 50 steps. In contrast to the previous notebook, we'll add a MonteCarloBarostat to fix the pressure, while allowing the volume to fluctuate. Our simulation corresponds to the $NPT$ ensemble.


```python
import openmm

TEMPERATURE = 300 * openmm.unit.kelvin
PRESSURE = 1 * openmm.unit.atmosphere
FRICTION_COEFFICIENT = 1 / openmm.unit.picosecond
TIMESTEP = 0.002 * openmm.unit.picoseconds

# Construct and configure a LangevinIntegrator at 300 K with an appropriate friction constant and time-step
integrator = openmm.LangevinIntegrator(
    TEMPERATURE,
    FRICTION_COEFFICIENT,
    TIMESTEP,
)
barostat = openmm.MonteCarloBarostat(PRESSURE, TEMPERATURE)

# Under the hood, this creates *OpenMM* `System` and `Topology` objects, then combines them together
simulation = interchange.to_openmm_simulation(integrator=integrator, additional_forces=[barostat])

# Add a reporter to record the structure every 50 steps
dcd_reporter = openmm.app.DCDReporter("trajectory.dcd", 50)
simulation.reporters.append(dcd_reporter)
```

### 4.2 Minimise the combined system

This will reduce any forces that are too large to integrate, such as from clashes or from disagreements between the crystal structure and force field.



```python
import numpy as np

def describe_state(state: openmm.State, name: str = "State"):
    max_force = max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces())
    print(
        f"{name} has energy {round(state.getPotentialEnergy()._value, 2)} kJ/mol "
        f"with maximum force {round(max_force, 2)} kJ/(mol nm)"
    )


describe_state(
    simulation.context.getState(
        getEnergy=True,
        getForces=True,
    ),
    "Original state",
)

simulation.minimizeEnergy()

describe_state(
    simulation.context.getState(getEnergy=True, getForces=True),
    "Minimized state",
)
```

    Original state has energy 14441273.68 kJ/mol with maximum force 1367208975.66 kJ/(mol nm)


    Minimized state has energy -436658.5 kJ/mol with maximum force 2427.9 kJ/(mol nm)


### 4.3 Run a short simulation

If this were anything more than a demonstration of the Toolkit, this example would need to include additional steps like equilibration. 

<div class="alert alert-warning" style="max-width: 700px; margin-left: auto; margin-right: auto;">
⚠️ Make sure you use your own, valid simulation protocol! This is just an example.
</div>


```python
simulation.context.setVelocitiesToTemperature(TEMPERATURE)
simulation.runForClockTime(1.0 * openmm.unit.minute)
```

_(This'll take a minute - literally, this time)_

While that runs, let's talk a bit about OpenFF.

### Open Source Force Fields

A primary goal of the Open Force Field Initiative is to make development and use of force fields as open as possible - it's in our name! We believe that open source development practices have a lot to offer the scientific community, whether that science is academic, commercial, or hobbyist.

#### The SMIRNOFF specification

The SMIRNOFF specification describes a simple format for describing molecular force fields. We provide and maintain this spec in the hopes that it will allow scientists everywhere to contribute to force field development in a unified way, without taking them away from their favourite simulation package.

SMIRNOFF is not just a spec; we're also committed to a reference implementation — that being the OpenFF Toolkit. The Toolkit endeavors to support all the functional forms in both the SMIRNOFF spec and the [`openff-forcefields`](https://github.com/openforcefield/openff-forcefields/) package.

#### Reproducibility

OpenFF force fields are completely specified by the name of the distributed `.offxml` file. We use codenames, version numbers, and tags to accomplish this. This means that as long as a user, designer, or reviewer sees the name of the force field being used, they know exactly what is going in to that simulation. We include parameters that are often neglected in force field specifications, such as the non-bonded cut-off distance, Ewald methods, constraints, modifications to the Lennard-Jones function, and partial charge generation methods are all defined by the name of the force field. 

As much as possible, we want energy and force to be a deterministic output of combining a molecule and a force field. If an author provides the name of the force field in their methods section, it should be reproducible. The other side of this coin is that we never want to hide the force field from the user. In all our workflows, the name of the force field must be explicitly provided in the code. This improves reproducibility of the code and helps the user take responsibility for their results. 

#### "Plugin" support for new force fields

The OpenFF Toolkit supports distributing force field files (.offxml) through Conda data packages. Anyone can publish a package on Conda Forge that extends the list of directories the toolkit searches for force fields, allowing anyone to produce force fields without requiring their own tooling, in a format that is designed to be converted to a multitude of simulation packages. See the [FAQ](https://open-forcefield-toolkit.readthedocs.io/en/stable/faq.html#how-can-i-distribute-my-own-force-fields-in-smirnoff-format) for more details.

---

Right! Simulation should be done by now, let's take a look.

### 4.4 Visualize the simulation with nglview

NGLView can display single structures and entire trajectories. Mouse over the widget to see the animation controls. Note that the trajectory generated above is `trajectory.dcd`. However, as this was run on CPUs (slow), we've provided a longer trajectory (`trajectory_gpu.dcd`, saved every 500 steps (1 ps)) for you to use for analysis. Feel free to visualise/ analyse both.


```python
import MDAnalysis as mda

u = mda.Universe(interchange.to_openmm_topology(), "trajectory_gpu.dcd")

view = nglview.show_mdanalysis(u)
view.add_representation("line", selection="protein")
view
```

    /opt/conda/envs/openff-env/lib/python3.14/site-packages/MDAnalysis/coordinates/DCD.py:171: DeprecationWarning: DCDReader currently makes independent timesteps by copying self.ts while other readers update self.ts inplace. This behavior will be changed in 3.0 to be the same as other readers. Read more at https://github.com/MDAnalysis/mdanalysis/issues/3889 to learn if this change in behavior might affect you.
      warnings.warn("DCDReader currently makes independent timesteps"



    NGLWidget(max_frame=265)


<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Can you visualize this trajectory in VMD (or another visualization tool of your choice)? Hint: DCD files only include the trajectory data (positions of each atoms over time) and lack topological information, so you might need another file.
</div>

<a id="analyse"></a>
## 5. MDAnalysis and ProLIF Enable Analysis of Pose Stability and Protein-Ligand Interactions

[`MDAnalysis`](https://www.mdanalysis.org/) is an excellent package (and an alternative to `MDTraj`) for the analysis of simulation data. [`ProLIF`](https://prolif.readthedocs.io/en/stable/) is a handy tool for computing protein-ligand interaction fingerprints based on `MDAnalysis` and the [`RDKit`](https://www.rdkit.org/docs/index.html). We'll demonstrate the use of each to perform some analyses specific to protein-ligand complexes. In particular, we'll analyse:

1. Binding pose stability
1. Protein-ligand interactions

### 5.1 Binding pose stability

RMSDs can tell us about the stability of our binding pose. We can calculate the RMSD of the ligand using `MDAnalysis`:


```python
from MDAnalysis.analysis import rms
import pandas as pd

# Define a selection string for the ligand
LIGAND = "resname LIG"

# Compute the RMSD of the ligand aligned to itself
R = rms.RMSD(u,  # universe to align
             u,  # reference universe or atomgroup
             select=LIGAND,  # group to superimpose and calculate RMSD
             ref_frame=0)  # frame index of the reference
R.run()

# Convert the results to a pandas DataFrame for easier handling
df = pd.DataFrame(R.results.rmsd,
                  columns=['Frame', 'Time (ps)', 'Ligand'])

# Plot
ax = df.plot(x='Time (ps)', y='Ligand', kind='line')
ax.set_ylabel(r'RMSD (Å)')
```




    Text(0, 0.5, 'RMSD (Å)')




    
![png](output_40_1.png)
    


<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Does the RMSD suggest any distinct ligand conformational states? Can you identify what these states correspond to from the trajectory?
</div>

<div class="alert alert-warning" style="max-width: 700px; margin-left: auto; margin-right: auto;">
⚠️ The above analysis tells us about ligand conformation, but is missing key information about pose stability.
</div>

<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Why is the above analysis not sufficient for determining pose stability? (Hint: what would happen if a rigid ligand drifted out of the binding site?) Can you modify the code above to compute an RMSD which is more informative about binding pose stability? (Hint: think about the alignment reference. You can specify additional groups which are used to calculate RMSD but not to align the trajectory -- have a look at the <a href="https://userguide.mdanalysis.org/stable/examples/analysis/alignment_and_rms/rmsd.html">MDAnalysis RMSD tutorial</a>. You might want to check out the <a href="https://userguide.mdanalysis.org/stable/selections.html"> MDAnalysis atom selection language documentation </a>.) When would you expect this new RMSD to be similar or different to the RMSD computed above? If you're stuck, you can click to reveal another hint below.
</div>

<details>
  <summary><b>Click here to reveal another hint</b></summary>
You'll likely want to use a selection string similar to <code>POCKET_RESIDUES = "protein and byres around 10.0 resname LIG"</code>
</details>


```python
POCKET_RESIDUES = "protein and byres around 10.0 resname LIG"

# Compute the RMSD of the ligand aligned to itself
R = rms.RMSD(u,  # universe to align
             u,  # reference universe or atomgroup
             select=POCKET_RESIDUES,  # group to superimpose and calculate RMSD
             groupselections=[LIGAND],
             ref_frame=0)  # frame index of the reference
R.run()

# Convert the results to a pandas DataFrame for easier handling
df = pd.DataFrame(R.results.rmsd,
                  columns=['Frame', 'Time (ps)', 'Pocket Residues', 'Ligand'])

# Plot
ax = df.plot(x='Time (ps)', y='Ligand', kind='line')
ax.set_ylabel(r'RMSD (Å)')
```




    Text(0, 0.5, 'RMSD (Å)')




    
![png](output_42_1.png)
    


<div class="alert alert-warning" style="max-width: 700px; margin-left: auto; margin-right: auto;">
⚠️ Be careful when interpreting RMSDs. A low RMSD is highly informative because it tells you the relevant structure is very similar to your reference, and there are few ways to be similar; a high RMSD is much less informative because it tells you the structures are different, and there are many ways to be different. <a href=https://pubs.acs.org/doi/10.1021/jp412776d>We should take care when interpreting stable high-RMSD conformations as well-defined stable states</a>. <a href="https://pubs.acs.org/doi/10.1021/acs.jctc.7b00028">The interpretation of RMSD may also be affected by molecular size</a>. 
</div>

### 5.2 Protein-Ligand Interactions

Let's analyse the protein-ligand interactions with `ProLIF`. This section is adapted from the [ProLIF tutorial](https://prolif.readthedocs.io/en/stable/notebooks/md-ligand-protein.html#ligand-protein-md) -- check it out for more analyses. First, let's select the ligand and binding site residues:


```python
import prolif as plf

POCKET_RESIDUES = "protein and byres around 10.0 resname LIG"

# create selections for the ligand and protein
ligand_selection = u.select_atoms(LIGAND)
protein_selection = u.select_atoms(POCKET_RESIDUES)
ligand_selection, protein_selection
```

... and visualise them:


```python
# create a molecule from the MDAnalysis selection
ligand_mol = plf.Molecule.from_mda(ligand_selection)
# display
plf.display_residues(ligand_mol, size=(400, 200))
```


```python
protein_mol = plf.Molecule.from_mda(protein_selection)
# remove the `slice(20)` part to show all residues
plf.display_residues(protein_mol, slice(20))
```

Now, let's calculate the interaction fingerprint using every 10th frame and specifying `count=True` to keep track of all interations (not just the first group of atoms that satisfied the constraints per interaction type and residue pair).

<div class="alert alert-info" style="max-width: 700px; margin-left: auto; margin-right: auto;">
    ℹ️ You may want to adjust the frame frequency depending on how many frames you generated.
</div>



```python
# use default interactions
fp = plf.Fingerprint(count=True)
# run on a slice of the trajectory frames: from begining to end with a step of 10
FRAME_FREQUENCY = 10 # Adjust this value as needed
fp.run(u.trajectory[::FRAME_FREQUENCY], ligand_selection, protein_selection)
```

`ProLIF` provides handy functions to display interactions over time, and to visualise the interactions in 2D and 3D!


```python
# Display interactions over time
fp.plot_barcode()
```


```python
# Plot the interactions in 2D
vis = fp.plot_lignetwork(ligand_mol)
vis
```


```python
# Plot interactions in 3D!
frame = 0

# Seek specific frame
u.trajectory[frame]
ligand_mol = plf.Molecule.from_mda(ligand_selection)
protein_mol = plf.Molecule.from_mda(protein_selection)

# Display
view = fp.plot_3d(ligand_mol, protein_mol, frame=frame, display_all=False)
view
```

<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Convert the fingerprint object to a pandas dataframe and list the residues by in order of the highest % interaction time. (Hint: check out the full <a href=https://prolif.readthedocs.io/en/stable/notebooks/md-ligand-protein.html#ligand-protein-md> ProLIF tutorial</a>. You might want to set <code>count=False</code> when calculating the fingerprint to get a percentage between 0 and 100.) What do these analyses tell us (and not tell us) about interaction strength?
</div>



```python
# Regenerate with count = False
fp = plf.Fingerprint(count=False)

# run on a slice of the trajectory frames: from begining to end with a step of 10
FRAME_FREQUENCY = 10 # Adjust this value as needed
fp.run(u.trajectory[::FRAME_FREQUENCY], ligand_selection, protein_selection)

# Add quantitative fingerprint analysis here...
df = fp.to_dataframe()
# percentage of the trajectory where each interaction is present
(df.mean().sort_values(ascending=False).to_frame(name="%").T * 100)
```

<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Repeat this entire notebook using a ligand docked to MCL-1. (Hint: You'll need to convert the pdbqt files to sdf files using obabel, adding protons as appropriate for pH 7. This will look something like <code>obabel docked_ligand.pdbqt -opdb | obabel -ipdb -osdf -p 7.0 -O docked_ligand.sdf</code>. Make sure to use the docked coordinates! An example docked pdbqt file is provided at <code>../structures/docked_ligand.pdbqt</code>) Is the binding pose stable? Are similar interactions formed by the docked ligand and the crystallographic ligand? Which do you think is likely to bind more strongly? What would be required to answer these questions robustly?
</div>


```python
# Run MD for a docked-ligand protein complex and analyse...
```

<a id="summary"></a>
## 6. Conclusions

* The OpenFF workflow cleanly separates the chemical system from its model.
* We parametrize ligands and proteins with the same software tools.
* Open source tools installed via Conda did everything, from system assembly to simulation, visualization, and analysis.
* Using OpenMM, we never had to leave Python to set up the simulation.
* With Interchange, using OpenMM, GROMACS, Amber or LAMMPS is simple!
* MDAnalysis and ProLIF allows us to perform varied analyses of our trajectories.
* The Rosemary force field is likely coming soon, and will allow easy set-up of simulations with post-translationally modified proteins. Check out [this workshop](https://github.com/openforcefield/2026-virtual-workshops/blob/main/ptm/ptm-workshop.ipynb)!

<a id="further_materials"></a>
## 7. There's Lots More to OpenFF!

A variety of example notebooks for OpenFF software are provided [here](https://docs.openforcefield.org/en/latest/examples.html). A few which are particularly relevant are:

- [Simulating Post-Translationally Modified Proteins with the OpenFF Rosemary Alpha](https://github.com/openforcefield/2026-virtual-workshops/blob/main/ptm/ptm-workshop.ipynb)
- [Host-guest systems](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-interchange/host-guest/host_guest.html)
- [Protein-ligand-water systems with Interchange](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-interchange/protein_ligand/protein_ligand.html). This has a lot of overlap with the current notebook, but there are several extra details not covered here.

<a id="further_non_openff_materials"></a>
## 8. Beyond OpenFF

You can parameterise your complex and run molecular dynamics -- so what's next? If you're interested in quantitatively assessing the binding affinity of your ligand for your target, then [alchemical (and path-based) free energy calculations are the gold-standard method](https://livecomsjournal.org/index.php/livecoms/article/view/v2i1e18378). [Open Free Energy](https://openfree.energy/) is another [Open Molecular Software Foundation](https://omsf.io/) initiative, which develops open-source tools for binding free energy calculations. However, these calculations are computationally demanding. If you're interested in a relatively fast (but relatively inaccurate) ranking of the binding affinities of a set of ligands, methods such as MM/GBSA may be appropriate. Affinity prediction methods based on deep learning such as Boltz-2 are appealingly fast, [but perform poorly on systems dissimilar to those they are trained on, and often show inflated performance on benchmarks due to data leakage](https://www.biorxiv.org/content/10.64898/2026.06.29.735309v1.abstract).
