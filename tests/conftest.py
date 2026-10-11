"""Register the hypothesis profiles a run may select.

Every ``@given`` test in this suite writes its own ``max_examples``, so that
a local or CI run takes a predictable time. The exception is the slow-marked
deep differential soaks, which take ``max_examples`` from whichever profile is
active here: the nightly job can then push them far deeper without editing a
test file per run. The "default" profile carries the same value those tests
would otherwise hardcode, so an unset ``HYPOTHESIS_PROFILE`` changes nothing.
"""

from __future__ import annotations

import os

from hypothesis import settings

settings.register_profile("default", max_examples=1000, deadline=None)
settings.register_profile("nightly", max_examples=5000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
