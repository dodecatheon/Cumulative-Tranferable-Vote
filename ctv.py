#!/usr/bin/env python
"""\
Cumulative Transferable Voting
Copyright (C) 2010-2011, Louis G. "Ted" Stern

Provide a Python class to run a cumulative transferable voting election on a
set of ballots, in order to implement proportional representation.

The score of each ballot is normalized to sum to 1.

Standard Cumulative Voting (CV) simply drops the candidates with the lowest
total scores until only the number to be seated still remains.

With CTV, a ballot may have some portion of the score from over-votes
or elimination applied to other candidates on the same ballot.

The difference shows up mainly in the last one or two seats, or the
last seat of a multi-seat faction, where vote-splitting or over-vote
waste could have an effect.

With Hare quota (total / n_seats), voting blocks get as much representation
as they have, with the remainder lost or transfered to alternate party
candidates.  If there are votes left over, they are applied, if applicable,
to alternate party candidates with small residual votes on each ballot.

With Hare, the effective quota for the last seat is actually N /
(2*M), therefore Hare also gives more representation to smaller
parties.

The Droop quota (int(total / n_seats) + 1) guarantees that a majority
party will win a majority of the seats.

Larger political parties would therefore prefer Droop to maximize
their factional strength.

With 7 or more seats, the difference in effect between the quotas is
not as significant as with 5 or fewer, and small parties have more
liklihood of achieving representation.  But it may be more difficult
for voters to choose 7 or more candidates in a crowded field.

Glossary of terms, following the analogy of "musical chairs":

Eliminated:  candidate was removed from active list.

Seated:      candidate has met quota or survived elimination to become a winner.

Standing:    candidate has neither met quota or been eliminated and is
             still in the running.

Locked:      a candidate's vote becomes locked on a single ballot when it
             cannot be transfered to any other candidate in the event
             of an overquota rescaling or elimination.

             If candidate A's score is locked on a ballot on the first
             round, it is usually called a bullet vote -- the voter
             picked a single candidate.

             But it can also happen when all other choices on the
             ballot have already been seated or eliminated.  Then if
             candidate A receives an overvote or is eliminated, the
             ballot's excess score for A is not transferable.

Locksums:    The sum of locked normalized scores for a candidate determine
             how much of the candidate's total score can be rescaled and
             redistributed in the event of being seated with an overquota.

Support:     Indicates whether a candidate received non-zero score on a ballot.

"""
# For faster reverse sorting (in 2.4+):
from operator import itemgetter
from textwrap import dedent
import os


# Utility function to sort a dictionary in reverse order of values
def _reverse_sort_dict(d):
    return sorted(d.iteritems(), key=itemgetter(1), reverse=True)

# Utility function to print out candidate:score tuples in reverse sorted order:
def _reverse_print_tuples(tlist):
    for t in sorted(tlist, key=itemgetter(1), reverse=True):
        print"\t%s: %15.6f" % t
    print "\n"

# shared dict of constants for different quota types
_qconst = {'hare':0.0, 'droop':1.0}

# Set up for Hare or Droop quotas:
def _calc_quota(n, nseats=5, qtype='hare'):

    # Hare quota = Nvotes / Nseats
    # Droop quota = modified version of int(Nvotes / (Nseats + 1)) + 1
    #
    # Since CTV is based on CV, calculate Droop using the usual
    # Cumulative Voting style in which each voter gets Nseats votes:
    #
    #  multiply number of voters * number of seats, this is like
    #    standard CV giving each voter "nseats" votes.
    #  divide by number of seats + one
    #  round down
    #  add one
    #  Finally, divide by nseats to get fraction of original votes.
    #
    #  With Droop, it is like rounding up to the nearest 1/nseats of a vote.
    #  By using _qconst[qtype], this formula works for Hare also.
    #
    return (float(int(float(n*nseats)/(float(nseats)+_qconst[qtype])))
            +_qconst[qtype])/float(nseats)

def _maxmin_dict_items(totals, standing):
    maxkey = None
    maxval = 0.0
    minkey = None
    minval = 2.e9
    for k, v in totals.iteritems():
        if k in standing:
            if v > maxval:
                maxkey = k
                maxval = v
            if v < minval:
                minkey = k
                minval = v

    return (maxkey, maxval, minkey, minval)

class CtvElection(object):

    def __init__(self,
                 ballots=[],
                 filename=None,
                 qtype='hare',
                 nseats=5):
        "Initialize from a list of ballot dicts or a filename"

        # Number of seats to fill:
        self.nseats = nseats

        # Quota type
        self.qtype = qtype

        # no quota, initially
        self.quota = None

        # Not normalized, initially
        self.normalized = False

        # List of candidates:
        self.candidates = set([])

        # List of factors:
        #  [ ('cand1', gamma1), ('cand2',gamma2), ... ]
        self.gammas = []

        # List of elected candidates
        self.seated = set([])
        self.ordered_seated = []

        # List of eliminated candidates
        self.eliminated = set([])

        # Candidates neither seated or eliminated
        # (as in musical chairs)
        self.standing = set([])

        # List of totals as we go through over-vote transfer and eliminations
        self.totals_list = []

        # List of locked sums to go along with total_list
        self.locksums_list = []

        # List of lists of SUPPORT
        self.support_list = []

        # ------------------------------------------------------------
        # Absorb ballots, either from input list or from a file

        if ballots:
            self.ballots = ballots
        else:
            self.ballots = []

        if filename:
            if os.path.splitext(filename)[1] == '.csv':
                self.csv_ballots(filename)
            else:
                self.append_ballots(filename)

        self.nvotes = len(self.ballots)

        if qtype not in _qconst.keys():
            print "Error, qtype not recognized"
            sys.exit(1)

    def csv_ballots(self,
                    filename):
        """\
        Read ballots from a csv file.
        First line is names of candidates."""
        f = open(filename,'r')
        keys = f.readline().rstrip().split(',')
        for line in f:
            ballot = {}
            for i, v in enumerate(line.rstrip().split(',')):
                if v:
                    if int(v):
                        ballot[keys[i]] = float(int(v))
            self.ballots.append(ballot)

        f.close()

        self.nvotes = len(self.ballots)


    def append_ballots(self,
                       filename):
        """We assume each line 1of file looks like 'a:#, b:#, c:'.
I.e., values separated by comma-space, with key:val pairs separated by
colons"""
        f = open(filename,'r'); lines = f.readlines(); f.close()
        print "number of lines read = ",
        print len(lines)
        for line in lines:
            self.ballots.append(dict([(pair.split(':')[0].lower(),
                                       float(pair.split(':')[1]))
                                      for pair in line.rstrip().split(', ')]))
        self.nvotes = len(self.ballots)

    def set_quota(self, qtype=None):
        if qtype:
            self.qtype = qtype

        self.quota = _calc_quota(self.nvotes,
                                 nseats=self.nseats,
                                 qtype=self.qtype)

        print "Using %s quota" % self.qtype.capitalize()

        print "Setting quota to %g out of %g" % (self.quota, self.nvotes)

        print "Quota percentage = %g %%" % (float(self.quota)
                                         / self.nvotes * 100.00)

        print ""

    def print_ballots(self):
        for i, ballot in enumerate(self.ballots):
            print "ballots[%d] = {" % i,
            for key, val in ballot.iteritems():
                print "'%s':%f, " % (key, val * self.nseats),
            print "}"

    def print_totals(self):
        "Print vote totals in terms of quota percentages"
        for i, totals in enumerate(self.totals_list):
            im1 = i - 1
            if i > 0:
                (c, gamma) = self.gammas[im1]
                if gamma >= 0.0:
                    print "\nCandidate %s weighted by gamma = %g" % \
                        (c, gamma)
                else:
                    print "\nCandidate %s was seated without vote transfer" % \
                        c

            print "totals[%d] = {" % i,
            for key, val in _reverse_sort_dict(totals):
                print "'%s':%6.2f%%, " % (key, val*100.00/self.quota),
            print "}"

    # Candidate "support" is simply the number of votes of any strength
    # for that candidate.
    def calc_support(self):
        support = {}
        for ballot in self.ballots:
            for c in ballot.keys():
                n = support.get(c,0)
                support[c] = n + 1

        return _reverse_sort_dict(support)

    def normalize(self):
        # Normalize, return initial totals and ballots:
        totals = {}
        locksums = {}
        support = {}

        self.nvotes = len(self.ballots)

        # copy ballots array to ensure no pointer duplication,
        # and remove zero scores.
        ballots = [dict([(c, score)
                         for (c, score) in ballot.items()
                         if score > 0.0])
                   for ballot in self.ballots]

        # Operate on the copy
        for ballot in ballots:

            vals = ballot.values()
            SUM = sum(vals)
            n = len(vals)

            for c, score in ballot.iteritems():
                score /= SUM
                ballot[c] = score
                totals[c] = totals.get(c,0.0) + score
                ns = support.get(c,0)
                support[c] = ns + 1
                if n == 1:
                    locksums[c] = locksums.get(c,0.0) + score

        support_list = _reverse_sort_dict(support)

        # Accumulate the reverse sorted support_list
        self.support_list.append(support_list)

        # Make the copy the new ballots list
        self.ballots = ballots

        self.normalized = True
        self.candidates = set(totals.keys())
        self.standing = self.candidates
        self.totals_list.append(totals)
        self.locksums_list.append(locksums)
        self.set_quota()

    # Cumulative transfer function:
    def transfer_votes(self, cand, gamma, eliminate=False):
        """\
The score of candidate 'cand' is adjusted by factor 'gamma' on all
ballots."""
        totals = {}
        locksums = {}
        support = {}

        for ballot in self.ballots:

            # Intersect the current ballot's choices with all 'standing'
            # candidates, those who have been neither seated or eliminated

            # Set of all choices on this ballot:
            choices = set(ballot.keys())

            # Set of all non-seated candidates on this ballot, excluding
            # 'cand':
            standing = choices & self.standing - set([cand])

            # size of the set of non-seated candidates on this ballot:
            n = len(standing)

            # If 'cand' is on this ballot and there are other candidates
            # to transfer to, ...
            if ((n>0) and
                (cand in choices)):
                # Transfer only if there are some unfinished on the ballot
                # and the selected candidate 'cand' is on the ballot

                # extract current score for 'cand' and remove from ballot:
                beta = ballot.pop(cand, 0.0)

                # If new score is non-zero, put it back on the ballot:
                newbeta = beta * gamma
                if newbeta > 0.0:
                    ballot[cand] = newbeta

                # Scaling coefficient for unfinished:
                # Divide each of the non-finished' scores by S,
                # Then multiply by S plus the amount that was taken away
                # from candidate 'cand'
                S = sum(ballot[c] for c in standing)
                phi = (S + (1.0 - gamma)*beta) / S
                for c in standing:
                    ballot[c] *= phi

            # If there is only one standing candidate on this ballot,
            # keep track of the non-transferable "locked" vote for
            # calculating scaling factor "gamma" in the next round:
            if n == 1:
                for c in standing:
                    locksums[c] = locksums.get(c, 0.0) + ballot[c]

            # Fo eliminations, we delete the candidate from the ballot:
            if eliminate:
                beta = ballot.pop(cand, 0.0)

            # Accumulate all totals from this ballot regardless
            # of whether candidates are seated or standing
            for c, score in ballot.iteritems():
                totals[c] = totals.get(c, 0.0) + score
                ns = support.get(c,0)
                support[c] = ns + 1

        # Keep track of the list of candidates with their voter support
        support_list = _reverse_sort_dict(support)

        # Accumulate the reverse sorted support_list
        self.support_list.append(support_list)

        # Update lists:
        self.gammas.append((cand, gamma))
        self.totals_list.append(totals)
        self.locksums_list.append(locksums)

    def run_election(self):
        if not self.normalized:
            self.normalize()

        totals = self.totals_list[-1]
        locksums = self.locksums_list[-1]

        print "Starting with totals = "
        initial_totals = _reverse_sort_dict(totals)
        _reverse_print_tuples(initial_totals)

        # Keep track of the CV winning set
        cv_winning_set = set([c
                              for c, score in initial_totals[0:self.nseats]])

        print "Starting with locksums = ", locksums
        print ""

        while len(self.seated) < self.nseats:
            totals = self.totals_list[-1]
            locksums = self.locksums_list[-1]
            support_list = self.support_list[-1]

            (maxkey,
             maxval,
             minkey,
             minval) = _maxmin_dict_items(totals, self.standing)

            # "locksum" is the total vote that cannot be transfered
            # to other candidates.
            locksum = locksums.get(maxkey,0.0)

            if ((maxval == self.quota) or
                ((maxval > self.quota) and
                 (maxval <= locksum))) :

                print "Candidate %s seated with no transfer" % maxkey
                if (locksum >= self.quota):
                    print "Transfer not possible because locksum is >= quota:"
                    print "Over-vote loss of %g votes --" % \
                        (locksum - self.quota)
                    print "no standing candidates left on %s ballots" % \
                        maxkey,
                    print "to transfer votes to"
                print ""

                # Candidate maxkey is a winner, and does not require
                # any transfers, or because of insufficient vote
                # distribution on ballots, the vote cannot be transfered.

                self.seated.add(maxkey)
                self.ordered_seated.append((maxkey,-1.0))
                self.standing.remove(maxkey)

                self.totals_list.append(totals)
                self.locksums_list.append(locksums)
                self.support_list.append(support_list)

                # Append a fake gamma to show no transfer:
                self.gammas.append((maxkey,-1.0))

            elif maxval > self.quota:
                # Candidate maxkey is a winner!
                # Redistribute excess votes.

                if locksum > self.quota:
                    print "Over-vote loss of %g votes --" % \
                        (locksum - self.quota)
                    print "No standing candidates left on %s ballots " % \
                        maxkey,
                    print "to transfer votes to\n"

                print "Candidate %s seated, scaled by" % maxkey,

                self.seated.add(maxkey)
                self.standing.remove(maxkey)

                # if maxval <= locksum: handled above
                gamma = (self.quota - locksum) / (maxval - locksum)

                if gamma < 0.0:
                    gamma = 0.0

                self.ordered_seated.append((maxkey,gamma))
                print "gamma = %g\n" % gamma

                print "Error check:  maxval = %g, totals[%s] = %g" % (
                    maxval,
                    maxkey,
                    totals[maxkey] )

                print "Error check:  gamma*(maxval-locksum)+locksum = ", \
                    gamma * (maxval-locksum) + locksum

                self.transfer_votes(maxkey, gamma)

                print "Error check:  after transfer, totals[%s] = %g" % (
                        maxkey,
                        self.totals_list[-1].get(maxkey,0.0) )

                print ""

            else:
                # Candidate minkey is a loser:
                # Eliminate candidate with smallest number of votes

                locksum = locksums.get(minkey,0.0)

                if minval > locksum:

                    print "Candidate %s eliminated with vote transfer" % minkey
                    print ""
                else:
                    print \
                        "Candidate %s eliminated, no transfer, lost vote = %15.6f", \
                        (minkey, locksum)
                    print ""

                self.eliminated.add(minkey)
                self.standing.remove(minkey)
                self.transfer_votes(minkey, 0.0, eliminate=True)

                # Ensure eliminated candidates are removed from totals
                # and locked sums
                elim_total  = self.totals_list[-1].pop(minkey,0.0)
                elim_locked = self.locksums_list[-1].pop(minkey,0.0)

            new_totals = self.totals_list[-1]
            new_locksums = self.locksums_list[-1]
            new_support_list = self.support_list[-1]

            totals_diff = []
            for c, score in new_totals.iteritems():
                diff = score - totals[c]
                if diff > 0.0:
                    totals_diff.append((c,diff))

            print "Transfer totals = "
            _reverse_print_tuples(totals_diff)

            # Totals, reverse sorted (descending order):
            rsort_totals = _reverse_sort_dict(new_totals)
            sort_totals = sorted(rsort_totals, key=itemgetter(1))

            n_seated = len(self.seated)
            n_needed = self.nseats - n_seated
            n_standing = len(self.standing)
            print "# of seated   candidates = %d" % n_seated
            print "# of standing candidates = %d" % n_standing
            print "# of open seats          = %d" % n_needed, "\n"

            if n_needed == 0:
                print "All seats have filled."
                print "Deleting all remaining standing candidates"
                for c in [k
                          for k, v in sort_totals
                          if k in self.standing]:
                    self.eliminated.add(c)
                    self.standing.remove(c)
                    dummy_c = new_totals.pop(c,0.0)
                    dummy_b = new_locksums.pop(c,0.0)
                    for i, supporttuple in enumerate(new_support_list):
                        if supporttuple[0] == c:
                            dummy_r = new_support_list.pop(i)
                            break
                rsort_totals = _reverse_sort_dict(new_totals)

            elif n_needed == n_standing :
                # No one left to transfer to on later steps,
                # so simply seat remaining candidates
                print "No more transfer candidates available"
                print "Seating all surviving candidates (%d needed == %d standing)" % ( n_needed, n_standing)
                print ""

                for c in [k
                          for k, v in rsort_totals
                          if k in self.standing]:
                    print "Candidate %s seated in final set of surviving candidates" % c
                    self.seated.add(c)
                    self.ordered_seated.append((c,-2.0))
                    self.standing.remove(c)
                    self.totals_list.append(new_totals)
                    self.locksums_list.append(new_totals)
                    self.support_list.append(new_support_list)
                    self.gammas.append((c,-1.0))

            print "After transfer, totals  = "
            _reverse_print_tuples(rsort_totals)

            print "After transfer, locksums = ", new_locksums
            print ""

        print "======================================"
        winners = _reverse_sort_dict(self.totals_list[-1])

        ctv_winning_set = set([c
                               for c, score in winners])

        print "Winners, in order seated = "
        for c, gamma in self.ordered_seated:
            if gamma >= 0.0:
                print "\t%s seated with score scaled by %8.6f" % (c, gamma)
            elif gamma == -1.0:
                print "\t%s seated with no transfer possible" % c
            elif gamma == -2.0:
                print "\t%s seated in final set" % c

        print "\nWinner total scores, in descending order:"
        _reverse_print_tuples(winners)

        # print "Individual totals = ", _reverse_sort_dict(new_totals)
        print "Sum of all totals = ", sum(new_totals[c] for c in self.seated)

        s = 0.0
        for v in new_totals.values():
            if v > self.quota:
                s += self.quota
            else:
                s += v
        print "Sum of untransferable votes = ", (self.nvotes - s)
        print "======================================"

#         print """
#
# Compact list of totals as transfers proceeded,
# represented as percentage of quota:
# """
        # self.print_totals()

        # print "\nSort initial totals to show traditional CV winners:\n"
        # print _reverse_sort_dict(self.totals_list[0])[0:self.nseats]
        #
        # print "\nWinners:"
        # print _reverse_sort_dict(self.totals_list[-1])

        print "\nInitial support list:"
        print self.support_list[0]

        print "\nFinal support list:"
        print self.support_list[-1]

        print "\nCV winners not in final CTV set"
        print list(cv_winning_set - ctv_winning_set)

        print "\nCTV winners not in initial top %d cumulative votes:" % self.nseats
        print list(ctv_winning_set - cv_winning_set)

        # print "\nFinal ballots, rescaled:"
        # for i, ballot in enumerate(self.ballots):
        #     m = max(ballot.values())
        #     print "Voter %d: " % (i+1),
        #     for k in xrange(101,117):
        #         kk = str(k)
        #         vv = ballot.get(kk,0.0)/m * 9.0
        #         if vv:
        #             print " C%s: %6.4f," % (kk,vv),
        #     print "\n",

if __name__ == "__main__":

    # Test with 100 ballots
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
        [{'a1':10., 'a2':9.0, 'a3':8.0, 'a4':7.0, 'a5':1.0, 'a6':1.0, 'c2':1.0},
         {'a1':10., 'a2':10., 'a3':10., 'a4':10., 'a5':1.0, 'a6':1.0, 'c3':1.0},
         {'a1':1.0, 'a2':10., 'a3':9.0, 'a4':8.0, 'a5':7.0, 'a6':1.0},
         {'a1':1.0, 'a2':10., 'a3':10., 'a4':10., 'a5':10., 'a6':1.0},
         {'a1':1.0, 'a2':1.0, 'a3':10., 'a4':9.0, 'a5':8.0, 'a6':7.0},
         {'a1':1.0, 'a2':1.0, 'a3':10., 'a4':10., 'a5':10., 'a6':10.},
         {'b1':10., 'b2':9.0, 'b3':8.0, 'b4':0.0, 'b5':0.0},
         {'b1':10., 'b2':10., 'b3':10., 'b4':0.0, 'b5':0.0},
         {'b1':0.0, 'b2':10., 'b3':9.0, 'b4':8.0, 'b5':0.0},
         {'b1':0.0, 'b2':10., 'b3':10., 'b4':10., 'b5':0.0},
         {'b1':0.0, 'b2':0.0, 'b3':10., 'b4':10., 'b5':10.}] + \
         5 * [{'c3':1.0, 'c1':10., 'c2':5.0},
              {'c1':1.0, 'c2':10., 'c3':7.0},
              {'c2':1.0, 'c3':10., 'c1':9.0}] + \
              8 * [{'d1':10.}]
    # ballots = \
    #     50 * [{'Z':99., 'X':42., 'Q':0.}] + \
    #     50 * [{'X':99., 'Q':43., 'Z':0.}] + \
    #     40 * [{'Q':99., 'Z':53., 'X':0.}] + \
    #     01 * [{'Q':99., 'X':77., 'Z':0.}]

    # ctv = CtvElection(ballots=ballots,
    #                   nseats=5,
    #                   qtype='hare'
    #                   )

    ctv = CtvElection(nseats=9,
                      filename='june2011.csv',
                      qtype='hare'
                      )

    ctv.run_election()