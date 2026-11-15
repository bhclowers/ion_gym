"""
ion_gym's warning category: a configuration that is legal but ill-advised,
or a fallback ion_gym took on the deck's behalf. Raised through Python's
warnings module, so the standard controls apply:

    import warnings
    from ion_gym.advisories import IonGymAdvisory
    warnings.simplefilter("always", IonGymAdvisory)   # every repeat
    warnings.simplefilter("error", IonGymAdvisory)    # stop on the first
    with warnings.catch_warnings(record=True) as seen: ...  # collect them

Under Python's default filter each distinct text is shown once per place
it is raised from, so a sweep that builds the same deck many times states
an advisory once, not once per point. The full list for a deck is always
available from SimSpec.advisories(), which never deduplicates.
"""
__all__ = ["IonGymAdvisory", "advise"]

import warnings


class IonGymAdvisory(UserWarning):
    """A legal but ill-advised configuration, or a fallback ion_gym took;
    the text states the reason and what was done."""


def advise(text, stacklevel=1):
    """Raise `text` as an IonGymAdvisory. stacklevel counts from the
    caller of advise (1 = the line that called advise)."""
    warnings.warn(str(text), IonGymAdvisory, stacklevel=stacklevel + 1)
