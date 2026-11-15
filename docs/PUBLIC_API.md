# The public API

ion_gym's public interface is the `ion-gym` command line and the names in
the `__all__` of the modules below. Those names are the ones meant for your
scripts and notebooks: each is documented in its module and exercised by
the test suite.

Everything else is internal. That includes every module not listed here,
any name not in a listed module's `__all__`, and any name beginning with an
underscore. Internal names can change in any release without notice.

ion_gym has not reached a 1.0 release, so the public names can still
change between versions too. When one does, the change is described with
the release that makes it.

| Module | What it is for |
|---|---|
| `ion_gym.io.sim_spec` | The deck: `SimSpec` and its parts (`GeometrySpec`, `SourceSpec`, `CollisionSpec`, `IntegrationSpec`, ...), and the recording and travelling-wave helpers |
| `ion_gym.io.spec_io` | `load_any_spec`: open a deck, a 3-D scene or an assembly from JSON |
| `ion_gym.physics.sim_build` | `build_run`: build a deck into a model and a fly function; `preflight`, the checks it makes first |
| `ion_gym.physics.ensemble_driver` | `run`, `fly_all`, `run_threaded`: fly an ensemble, with progress and stopping |
| `ion_gym.physics.flight_conditions` | `set_flight_conditions`: change voltages, drives, the ion or the gas without re-solving the field |
| `ion_gym.physics.stations` | Detector and recording planes: which ions crossed them, and where |
| `ion_gym.io.field_io` | Save, load, label and transfer solved fields and trajectory files |
| `ion_gym.io.results_bank` | Bank ensemble results as self-describing `.npz` archives |
| `ion_gym.io.run_provenance` | What a saved run needs in order to be flown again; `replay_spec` |
| `ion_gym.io.paths` | Where data, outputs, caches and the example decks live |
| `ion_gym.advisories` | `IonGymAdvisory`, the warning category for legal but ill-advised configurations |

## What `build_run` writes on the deck

`build_run(spec)` resolves derived values onto the `SimSpec` it is given,
and you can read them there afterwards: DC-ladder members' voltages, a
step derived from the drives when `dt_ns` is null, the field components a
derived channel needs, and (in random mode) the seed it drew. None of
these changes what a save of the deck writes, apart from ladder members'
voltages, which are derived from their group. Its docstring lists them.

## Command line

```
ion-gym --version
ion-gym dashboard [deck.json] [--port N] [--no-show]
ion-gym validate deck.json        # exit 0 valid, 1 refused, 2 unreadable
ion-gym edit [--port N] [--no-show]   # temporarily disabled: refuses, exit 2
```

`python -m ion_gym ...` is the same command.

## Advisories

A configuration that is legal but ill-advised is reported as an
`IonGymAdvisory` warning, so Python's warning controls apply:

```python
import warnings
from ion_gym.advisories import IonGymAdvisory

warnings.simplefilter("error", IonGymAdvisory)   # stop on the first one
```

`SimSpec.advisories()` returns the full list for a deck.
