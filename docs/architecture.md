# Architecture

```text
Process Streams
      |
      v
Validation + Units
      |
      v
Temperature Intervals
      |
      v
Heat Cascade
      |
      +--> Minimum Hot Utility
      +--> Minimum Cold Utility
      +--> Pinch Temperature
      |
      v
HEN calculations / network design

Separately, repeated exact cases feed:
      |
      v
Feature builder --> ANN (MLP) --> predicted targets
                         |
                         v
                  exact-vs-predicted evaluation
```

The ANN is a secondary prediction layer around the deterministic engineering model, not a replacement for physical validation.
