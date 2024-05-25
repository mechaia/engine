# Formats

## Intermediate Representation (IR)

### Data

All data types are sum types, which each variant having zero or more fields.

### Functions

All functions initially match on the arguments variants, or use a wilcard.
Each arm has a list of functions to call.

### Call model

All data is stored in registers.
Each function has its own sets of registers.
Since there is only one set of registers per function recursion is prohibited.
