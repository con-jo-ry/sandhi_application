# Sanskrit Sandhi Applier

Automatically applies sandhi rules to Sanskrit texts where sandhi has been dissolved and junction points are marked with `+` signs.

## Usage

```bash
python sandhi_applier_v3.py input.txt [sandhi_logic.txt]
```

### Arguments
- `input.txt` - Input file with dissolved sandhi (junctions marked with `+`)
- `sandhi_logic.txt` - *(Optional)* Custom rules file (defaults to `sandhi_logic.txt`)

### Output
- `input_sandhi.txt` - Processed text with sandhi applied
- `log.txt` - Detailed log of all transformations and unresolved junctions

### Testing
Run quick sanity tests:
```bash
python sandhi_applier_v3.py --test
```

## Rule Format

Rules are defined in `sandhi_logic.txt` using comma-separated values:

```
first_ending,second_beginning,result
```

### Examples
```
a,a,ā                    # a + a → ā
t,m,n m                  # t + m → n m (with space)
[a|ā],i,e                # (a or ā) + i → e
```

### Special Features

**Bracket notation** - Expands multiple options:
```
[a|ā|i],u,result         # Creates 3 rules: a+u, ā+u, i+u
```

**Preceding vowel context** - For word finals `ṅ`, `n`, `s`:
- The rule format includes the preceding vowel
- Example: `ān,a,n a` matches words ending in `-ān` before `a`

**Exceptional words** - Full-word matching for irregular forms:
```
# Exceptions
punas,a,punar a          # Word "punas" uses special rules
sas,k,sa k
```

Place exceptional word rules below the `# Exceptions` marker in the rules file.

## Input Format

Mark sandhi junctions with `+` (no spaces around the plus sign):

```
deva+atra              → devātra
tattvatas+punas+api    → tattvataḥ punar api
```

## Log File

The `log.txt` file records all changes:

```
tattvatas+punas+ → tattvataḥ punar
devā+atra → devātra
tattve+devatā → NO RULE FOUND (e+d)
```

Unresolved sandhi junctions are flagged as `NO RULE FOUND` with the pattern shown in parentheses.

## Requirements

- Python 3.7+
- No external dependencies (uses only standard library)

## Notes

- Rules are currently based on standard Sanskrit sandhi charts and may be verbose
- UTF-8 encoding required for both input and rules files
