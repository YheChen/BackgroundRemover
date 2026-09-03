"""The six stages, in pipeline order.

  1. classify      subject routing              -> stages/classify.py   (step 5)
  2. segment       coarse binary-ish mask       -> stages/segment.py
  3. trimap        uncertainty band             -> stages/trimap.py
  4. matte         continuous alpha in the band -> stages/matte.py
  5. decontaminate true foreground colour       -> stages/decontaminate.py
  6. composite     reproject to full res, output-> stages/composite.py

Stages 4 and 5 are where cutouts stop looking like cutouts. Stage 2 alone is
what produces the hard, halo-edged look around hair.
"""
