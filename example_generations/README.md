# Example generated duplex libraries

These files contain final sequence libraries produced by the Ortho pairwise
pre-synthesis screening workflow. Each line is one designed strand; its
cognate duplex partner is the reverse complement of that strand.

## Libraries

### `16nt_237_duplexes.txt`

- Length: 16 nt
- Final size: 237 duplexes (474 strand species)
- SeqWalk parameter: `k = 8`
- Maximum positional match count: 12
- On-target fraction threshold: 0.9
- Off-target fraction threshold: 0.1
- Operating condition: 1 uM per strand, 37 C, 1.0 M monovalent salt, 0 M Mg2+
- Tm grid: 25--84 C in 1 C increments
- Required library-wide pairwise Tm margin: 5 C
- Observed grid-estimated library-wide pairwise Tm margin: 16 C

Source: final sequence artifact from the complete 16-nt benchmark.

### `20nt_2361_duplexes.txt`

- Length: 20 nt
- Final size: 2,361 duplexes (4,722 strand species)
- SeqWalk parameter: `k = 9`
- Maximum positional match count: 12
- On-target fraction threshold: 0.9
- Off-target fraction threshold: 0.1
- Operating condition: 1 uM per strand, 47 C, 0.15 M monovalent salt, 0 M Mg2+
- Tm grid: 25--94 C in 1 C increments
- Required library-wide pairwise Tm margin: 5 C
- Observed grid-estimated library-wide pairwise Tm margin: 13 C

Source: final sequence artifact from the complete uncapped 20-nt scale-up.

Both benchmarks used NUPACK 4.0.2.0 with the DNA stacking ensemble. These
libraries satisfy the reported independent pairwise criteria; they are not a
claim that every strand species can coexist without globally coupled
competition.
