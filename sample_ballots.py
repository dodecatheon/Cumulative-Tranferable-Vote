#!/usr/bin/env python
# simple script to print out sample ballots in csv form

# Demo with 100 ballots
#
# Group A:  candidates a1, a2, a3, a4, a5, a6: 42%
# Group B:  candidates b1, b2, b3, b4, b5:     35%
# Group C:  candidates c1, c2, c3:             15%
# Group D:  candidate  d1:                      8%
#
# With Droop quota  and 5 seats,
# we'd expect
# Group A to get 2 seats,
# Group B to get 2 seats,
# and the last seat to be given to whoever avoids elimination the longest.
#
# If Group A wins the last seat, we have "lost"
#
#   Group B:  1%
#   Group C: 15%
#   Group D:  8%
#
# about 24 to 25%, or about 1/4 of the vote!
#
# If Group D wins the last seat, we lose
#   Group B:  1%
#   Group C: 15%
#   Group A:  8%
#
# About the same amount.
#
# But with CTV, one of the Group C candidates should accumulate 15% of
# the vote and beat any remaining A, B or D candidate, therefore we
# would lose only about
#   Group B:  1%
#   Group D:  8%
#   Group A:  8%
#
# or about 17%, about what you'd expect with a Droop quota.
# That's about 40% improvement in PR over standard CV.
#
ballots = \
    7 * \
    [{'a1':10, 'a2':9, 'a3':8, 'a4':7, 'a5':1, 'a6':1, 'c2':1},
     {'a1':10, 'a2':10, 'a3':10, 'a4':10, 'a5':1, 'a6':1, 'c3':1},
     {'a1':1, 'a2':10, 'a3':9, 'a4':8, 'a5':7, 'a6':1},
     {'a1':1, 'a2':10, 'a3':10, 'a4':10, 'a5':10, 'a6':1},
     {'a1':1, 'a2':1, 'a3':10, 'a4':9, 'a5':8, 'a6':7},
     {'a1':1, 'a2':1, 'a3':10, 'a4':10, 'a5':10, 'a6':10},
     {'b1':10, 'b2':9, 'b3':8, 'b4':0, 'b5':0},
     {'b1':10, 'b2':10, 'b3':10, 'b4':0, 'b5':0},
     {'b1':0, 'b2':10, 'b3':9, 'b4':8, 'b5':0},
     {'b1':0, 'b2':10, 'b3':10, 'b4':10, 'b5':0},
     {'b1':0, 'b2':0, 'b3':10, 'b4':10, 'b5':10}] + \
     5 * [{'c3':1, 'c1':10, 'c2':5},
          {'c1':1, 'c2':10, 'c3':7},
          {'c2':1, 'c3':10, 'c1':9}] + \
          8 * [{'d1':10}]

    # ballots = \
    #     50 * [{'Z':99, 'X':42, 'Q':0}] + \
    #     50 * [{'X':99, 'Q':43, 'Z':0}] + \
    #     40 * [{'Q':99, 'Z':53, 'X':0}] + \
    #     01 * [{'Q':99, 'X':77, 'Z':0}]

candidates = set([])

for ballot in ballots:
    for key in ballot.keys():
        candidates.add(key)

cands = sorted(list(candidates))
n = len(cands)
nm1 = n - 1

print ','.join(cands)
for ballot in ballots:
    print ','.join([str(ballot.get(c,''))
                    for c in cands])
