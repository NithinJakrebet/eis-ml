"""Dataset descriptions: which Ns steps hold the EIS sweep and the capacity label.

Every Bio-Logic protocol in ``data/`` is a sequence of numbered steps (``Ns``).
Which step carries the impedance sweep and which one carries the discharge
capacity differs per protocol, so each dataset is described by a small spec
that the feature builder reads. The EIS step(s) are *injectable*: pass
``eis_ns`` to :func:`get_dataset` (or ``--ns`` on the experiment CLI) to run
the same model on a different state of charge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

FREQ_RANGE_DEFAULT = (0.2, 20_000.0)


@dataclass(frozen=True)
class DatasetSpec:
    """Where to find EIS spectra and capacity labels inside one protocol.

    name:        folder under ``DATA_DIR`` (one CSV per cell inside it).
    eis_ns:      Ns step(s) whose impedance sweep forms the feature vector.
                 Several steps are concatenated in the order given.
    capacity_ns: discharge step whose ``Capacity/mA.h`` is the SOH label.
    freq_range:  keep frequencies ``lo < f <= hi`` (Hz); trims glitchy edges.
    cells:       restrict to these cells; ``None`` means every CSV in the folder.
    """

    name: str
    eis_ns: tuple[int, ...]
    capacity_ns: int
    freq_range: tuple[float, float] = FREQ_RANGE_DEFAULT
    cells: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.eis_ns:
            raise ValueError("eis_ns must name at least one Ns step")
        object.__setattr__(self, "eis_ns", tuple(int(n) for n in self.eis_ns))

    @property
    def tag(self) -> str:
        """Short id used for result folders, e.g. ``PEIS-HC-RT_ns1-6``."""
        return f"{self.name}_ns{'-'.join(map(str, self.eis_ns))}"

    def with_ns(self, *eis_ns: int) -> DatasetSpec:
        """Same protocol, different EIS step(s)."""
        return replace(self, eis_ns=tuple(eis_ns))


# Ns semantics per protocol (from the README protocol descriptions):
#   PEIS-HC-RT / GEIS-HC-RT : Ns=1 EIS before charge (discharged rest),
#                             Ns=6 EIS after charge (charged rest), Ns=8 CC discharge.
#   PEIS-HC-RT-sparseEIS    : Ns=5 EIS (every ~10 cycles), Ns=3 CC discharge.
#   Na_NoEIS                : no impedance columns at all; not usable here.
DATASETS: dict[str, DatasetSpec] = {
    "PEIS-HC-RT": DatasetSpec("PEIS-HC-RT", eis_ns=(1, 6), capacity_ns=8),
    "GEIS-HC-RT": DatasetSpec("GEIS-HC-RT", eis_ns=(1, 6), capacity_ns=8),
    "PEIS-HC-RT-sparseEIS": DatasetSpec("PEIS-HC-RT-sparseEIS", eis_ns=(5,), capacity_ns=3),
}


def get_dataset(
    name: str,
    eis_ns=None,
    capacity_ns: int | None = None,
    freq_range=None,
    cells=None,
) -> DatasetSpec:
    """Look up a preset by folder name and override any field.

    Unknown names are allowed as long as ``eis_ns`` and ``capacity_ns`` are
    given, so a new data folder needs no code change to be used.
    """
    base = DATASETS.get(name)
    if base is None:
        if eis_ns is None or capacity_ns is None:
            raise KeyError(
                f"Unknown dataset {name!r}; known: {sorted(DATASETS)}. "
                "Pass eis_ns and capacity_ns to describe a new one."
            )
        base = DatasetSpec(name, eis_ns=tuple(eis_ns), capacity_ns=capacity_ns)
    changes = {}
    if eis_ns is not None:
        changes["eis_ns"] = tuple(eis_ns)
    if capacity_ns is not None:
        changes["capacity_ns"] = int(capacity_ns)
    if freq_range is not None:
        changes["freq_range"] = (float(freq_range[0]), float(freq_range[1]))
    if cells is not None:
        changes["cells"] = tuple(cells)
    return replace(base, **changes) if changes else base
