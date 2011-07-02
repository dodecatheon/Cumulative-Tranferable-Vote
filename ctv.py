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
as they have, with the remainder lost or transferred to alternate party
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
             cannot be transferred to any other candidate in the event
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
from textwrap import fill, dedent
import re, os, sys
from math import log10


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
        # Absorb ballots, from input list and/or from file or stdin

        if ballots:
            self.ballots = ballots
        else:
            self.ballots = []

        if filename:
            if filename == '-':
                self.csv_ballots(stdin=True)
            else:
                self.csv_ballots(filename=filename)

        self.nvotes = len(self.ballots)

        if qtype not in _qconst.keys():
            print "Error, qtype not recognized"
            sys.exit(1)

    def csv_ballots(self,
                    filename=None, stdin=False):
        """\
Read ballots from a csv file.  First line is names of candidates."""

        if stdin:
            f = sys.stdin
        else:
            f = open(filename,'r')

        keys = f.readline().rstrip().split(',')
        for line in f:
            ballot = {}
            for i, v in enumerate(line.rstrip().split(',')):
                if v:
                    intv = int(v)
                    if intv:
                        ballot[keys[i]] = float(intv)
                        #ballot[keys[i]] = float(2.**(float(intv-1.0)/2.0))
                        #ballot[keys[i]] = log10(float(intv+1.0))
            self.ballots.append(ballot)

        if not stdin:
            f.close()

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
                    print "\nCandidate %s scaled by gamma = %g" % \
                        (c, gamma)
                else:
                    print "\nCandidate %s was seated with no vote transfer" % \
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

    def print_locksums(self, locksums):
        "Standard print function for locksums"
        locked_candidates = self.standing & set(locksums.keys())
        if len(locked_candidates) > 0:
            print "Current locked-vote totals:"
            for c in locked_candidates:
                print "\t%s: %15.6f" % (c, locksums[c])
        else:
            print "No votes locked"
        print ""

    def cross_correlation(self):
        # initial cross correlation dict of dicts
        cc = {}
        cands = list(self.standing)
        n = len(cands)

        if n <= 1:
            return

        for c1 in cands:
            cc[c1] = {}
            for c2 in cands:
                cc[c1][c2] = 0.0

        # add correlations for each ballot
        for ballot in self.ballots:
            cands = list(self.standing & set(ballot.keys()))
            n = len(cands)
            for i in xrange(n):
                ci = cands[i]
                vi = ballot.get(ci,0.0)
                if vi > 0.0:
                    cici = cc[ci].get(ci,0.0)
                    cc[ci][ci] = cici + vi * vi
                    if i < (n-1):
                        for j in xrange(i+1,n):
                            cj = cands[j]
                            vj = ballot.get(cj,0.0)
                            if vj > 0.0:
                                vivj = vi * vj
                                cicj = cc[ci].get(cj,0.0)
                                cjci = cc[cj].get(ci,0.0)
                                cc[ci][cj] = cicj + vivj
                                cc[cj][ci] = cjci + vivj

        # normalize correlations so that self-correlation = 1.0
        cands = list(self.standing)
        n = len(cands)

        for c1 in cands:
            v1sq = cc[c1][c1]
            for c2 in cands:
                c1c2 = cc[c1].get(c2,0.0)
                if c1c2 > 0.0:
                    cc[c1][c2] = c1c2 / v1sq
                else:
                    del cc[c1][c2]

        self.cc = cc


    def run_election(self, verbose=True, debug=False):
        "The meat of the method"
        if not self.normalized:
            self.normalize()

        # Asterisk denotes that candidate's seating is forced
        # because we've eliminated enough other candidates.
        asterisk = ''

        totals = self.totals_list[-1]
        locksums = self.locksums_list[-1]

        initial_totals = _reverse_sort_dict(totals)
        if verbose:
            print "Starting with totals:"
            _reverse_print_tuples(initial_totals)



        self.cross_correlation()
        if verbose:
            print "Cross correlations >= 0.1 indicate factions:"
            for c, score in initial_totals:
                print "%s:" % c, sorted([c2
                                         for c2, v2 in self.cc[c].iteritems()
                                         if c2 != c
                                         if v2 >= 0.1],
                                        key=itemgetter(0))
            print "\n",

        # Keep track of the CV winning set
        cv_winning_set = set([c
                              for c, score in initial_totals[0:self.nseats]])

        # Print initial locksum tallies
        if verbose:
            self.print_locksums(locksums)

        while len(self.seated) < self.nseats:
            totals = self.totals_list[-1]
            locksums = self.locksums_list[-1]
            support_list = self.support_list[-1]

            self.cross_correlation()

            (maxkey,
             maxval,
             minkey,
             minval) = _maxmin_dict_items(totals, self.standing)

            if verbose:
                print "Maximum and minimum scores:"
                print "\t%s: %15.6f" % (maxkey, maxval)
                print "\t%s: %15.6f" % (minkey, minval)
                print "\n",

            # "locksum" is the total vote that cannot be transferred
            # to other candidates.
            locksum = locksums.get(maxkey,0.0)

            if ((maxval == self.quota) or
                ((maxval > self.quota) and
                 (maxval <= locksum))) :

                # Transfer candidate
                tc = maxkey

                print "Candidate %s seated%s with no transfer, score = %15.6f" % \
                    (maxkey, asterisk, maxval)
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
                # distribution on ballots, the vote cannot be transferred.

                self.seated.add(maxkey)
                self.ordered_seated.append((maxkey,-1.0,asterisk))
                self.standing.remove(maxkey)

                self.totals_list.append(totals)
                self.locksums_list.append(locksums)
                self.support_list.append(support_list)

                # Append a fake gamma to show no transfer:
                self.gammas.append((maxkey,-1.0))

            elif maxval > self.quota:
                # Candidate maxkey is a winner!
                # Redistribute excess votes.

                # tc = Transfer candidate
                tc = maxkey

                if locksum > self.quota:
                    print "Over-vote loss of %g votes --" % \
                        (locksum - self.quota)
                    print "No standing candidates left on %s ballots " % \
                        maxkey,
                    print "to transfer votes to\n"

                self.seated.add(maxkey)
                self.standing.remove(maxkey)

                # if maxval <= locksum: handled above
                gamma = (self.quota - locksum) / (maxval - locksum)

                if gamma < 0.0:
                    gamma = 0.0

                self.ordered_seated.append((maxkey,gamma,asterisk))

                if debug:
                    print "Error check:  maxval = %g, totals[%s] = %g" % (
                        maxval,
                        maxkey,
                        totals[maxkey] )

                if debug:
                    print "Error check:  gamma*(maxval-locksum)+locksum = ", \
                        gamma * (maxval-locksum) + locksum

                self.transfer_votes(maxkey, gamma)

                print "Candidate %s seated%s," % (maxkey, asterisk), \
                    "scaled by gamma = %g," % gamma, \
                    "new score = %15.6f" % self.totals_list[-1].get(maxkey,0.0)

                print ""

            else:
                # Candidate minkey is a loser:
                # Eliminate candidate with smallest number of votes

                # tc = Transfer Candidate
                tc = minkey

                locksum = locksums.get(minkey,0.0)

                if minval > locksum:
                    print \
                        "Candidate %s eliminated," % minkey, \
                        "vote transferred = %15.6f\n" % minval
                else:
                    print \
                        "Candidate %s eliminated," % minkey, \
                        "cannot transfer, lost vote = %15.6f\n" % minval

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

            if verbose:
                if len(totals_diff) > 0 and verbose:
                    print "\t%-15s%18s%8s%18s" % ("Candidate",
                                                  "Transfer received",
                                                  "Xcorr",
                                                  "New score")
                    for c, s in sorted(totals_diff,
                                       key=itemgetter(1),
                                       reverse=True):
                        print "\t%-15s%18.6f%8.3f%18.6f" % ( c,
                                                             s,
                                                             self.cc[tc][c],
                                                             new_totals[c])
                    print "\n",

            # Totals, reverse sorted (descending order):
            rsort_totals = _reverse_sort_dict(new_totals)
            sort_totals = sorted(rsort_totals, key=itemgetter(1))

            n_seated = len(self.seated)
            n_needed = self.nseats - n_seated
            n_standing = len(self.standing)
            if verbose:
                print "# of seated   candidates = %d" % n_seated
                print "# of standing candidates = %d" % n_standing
                print "# of open seats          = %d" % n_needed, "\n"

            # Check new max and min scores:
            (maxkey,
             maxval,
             minkey,
             minval) = _maxmin_dict_items(new_totals,
                                          self.standing)

            if n_needed == 0:
                if verbose:
                    print "All seats are filled."
                    print "Deleting any remaining standing candidates"
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

            elif n_needed == n_standing:
                if maxval > self.quota:
                    if not asterisk:
                        asterisk = '*'
                        print 80*"*"
                        print fill(dedent("""\
                        We could seat all remaining candidates now
                        because the number of open seats is equal to the number
                        of surviving candidates.
                        """))
                        print "\n",
                        print fill(dedent("""\
                        However, we will continue transfering votes until
                        no candidate's total exceeds the quota, and will
                        indicate all subsequent forced seats with an
                        asterisk.
                        """))
                        print 80*"*"
                        print "\n",
                else:
                    # No more transfers possible,
                    # so simply seat remaining candidates
                    if verbose:
                        print "No more transfer candidates available"
                        print \
                            "Seating all surviving candidates", \
                            "(%d needed == %d standing)\n" % ( n_needed,
                                                             n_standing)
                    for c, v in rsort_totals:
                        if c in self.standing:
                            print \
                                "Candidate %s seated* in final set," % c, \
                                "vote = %15.6f" % v

                            self.seated.add(c)
                            self.ordered_seated.append((c,-2.0,asterisk))
                            self.standing.remove(c)
                            self.totals_list.append(new_totals)
                            self.locksums_list.append(new_totals)
                            self.support_list.append(new_support_list)
                            self.gammas.append((c,-1.0))
                    print ""

            # Print current running tally of locksums
            self.print_locksums(new_locksums)

        print 80*"="
        winners = _reverse_sort_dict(self.totals_list[-1])

        ctv_winning_set = set([c
                               for c, score in winners])

        print "Winners, in order seated:"
        for c, gamma, ast in self.ordered_seated:
            if gamma >= 0.0:
                print \
                    "\t%s seated with score scaled by %8.6f," % (c, gamma), \
                    "score = %15.6f%s" % (self.totals_list[-1][c], ast)
            elif gamma == -1.0:
                print \
                    "\t%s seated with no transfer possible,    " % c, \
                    "score = %15.6f%s" % (self.totals_list[-1][c], ast)
            elif gamma == -2.0:
                print \
                    "\t%s seated in final set,                 " % c, \
                    "score = %15.6f%s" % (self.totals_list[-1][c], ast)

        print "\n",

        if asterisk:
            print "*Selection inevitable after other candidates were eliminated.\n"

        # print "Individual totals = ", _reverse_sort_dict(new_totals)
        print "Sum of all totals = ", sum(new_totals[c] for c in self.seated)

        s = 0.0
        for v in new_totals.values():
            if v > self.quota:
                s += self.quota
            else:
                s += v
        print "Sum of untransferable votes = ", (self.nvotes - s)
        print 80*"="

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

        if debug:
            print "\nInitial support list:"
            print self.support_list[0]

            print "\nFinal support list:"
            print self.support_list[-1]

        cv_minus_ctv = cv_winning_set - ctv_winning_set
        ctv_minus_cv = ctv_winning_set - cv_winning_set

        if len(cv_minus_ctv) > 0:
            print "\nCV winners not in final CTV set: ", list(cv_minus_ctv)

            print "CTV winners not in initial top %d cumulative votes: " % \
                self.nseats, list(ctv_minus_cv)
        else:
            print "\nCTV winners are identical to CV winners."

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
    from optparse import OptionParser

    usage="""\
Usage %prog \\
            [-n|--nseats NSEATS] \\
            [-q|--quota-type QTYPE] \\
            [-f|--filename FILENAME.csv] \\
            [-v|--verbose] \\
            [-D|--debug]

%prog is a script to run Cumulative Transferable Voting (CTV) to
implement a Proportional Representation (PR) election, using a set of
tabulated ballots with ratings for each candidate.

The Comma Separated Variable format is assumed to be in the form

	name1,name2,name3,...
        ,,,,,9,,,6,,7,,,2,...
        ,,9,8,7,6,1,2,0,...

with the list of candidates on the first line, and non-zero scores
for the respective candidates as ballots on following lines.

On each ballot, scores are summed and then each score is divided by
the total.  This normalizes the ballot so all scores add up to 1,
giving each ballot a single vote.

To determine NSEATS winners, a quota is determined as NBALLOTS / NSEATS.

We iterate until all seats have been filled:

   Find max and min total scores.

   If the max score exceeds the quota, transfer excess votes on each ballot
   to standing candidates, in proportion to their score on that ballot.

   If max score does not exceed the quota, delete the candidate with min
   score and transfer vote to standing candidates on that ballot.

   Stop when

     The number of standing candidates equals the number of empty seats

     and

     No standing candidate has an over-quota vote to transfer.
"""
    version = "Version: %prog 0.1"

    parser = OptionParser(usage=usage, version=version)

    parser.add_option('-n',
                      '--nseats',
                      type=int,
                      default=7,
                      help=fill(dedent("""\
                      Number of winning seats for election.  [Default: 7]""")))

    parser.add_option('-q',
                      '--quota-type',
                      type='string',
                      default='hare',
                      help=fill(dedent("""\
                      Quota type used in election.  'hare' = Hare =
                      Number of ballots divided by number of seats.
                      'droop' = Droop = approximately Nballots /
                      (Nseats + 1), adjusted slightly.  [Default:
                      hare]""")))

    parser.add_option('-f',
                      '--filename',
                      type='string',
                      default='-',
                      help=fill(dedent("""\
                      Filename of comma-separated-variable (csv) file
                      containing ballots.  Use hyphen ('-') to take
                      input from stdin.  [Default: -]""")))

    parser.add_option('-v',
                      '--verbose',
                      action='store_true',
                      default=False,
                      help="Turn on verbose mode printout.  [Default:  False]")

    parser.add_option('-D',
                      '--debug',
                      action='store_true',
                      default=False,
                      help="Turn on debug mode printout.  [Default:  False]")

    opts, args = parser.parse_args()

    if not re.match(r"hare|droop",opts.quota_type):
        print "\nError, argument to --quota-type can be only 'hare' or 'droop'\n"
        parser.print_help()
        sys.exit(1)

    if (opts.nseats < 1):
        print "\nError, --nseats argument must be a positive integer\n"
        parser.print_help()
        sys.exit(1)

    if (opts.filename == "-"):
        print "Reading CSV input from stdin\n\n"
    else:
        if not os.path.isfile(opts.filename):
            print "\nError, %s file does not exist\n" % opts.filename
            parser.print_help()
            sys.exit(1)

        ext = os.path.splitext(opts.filename)[1]

        if ((ext != '.csv') and (ext != '.CSV')):
            print "\nError, %s file does not have .csv or .CSV extension\n"
            parser.print_help()
            sys.exit(1)

    ctv = CtvElection(nseats=opts.nseats,
                      filename=opts.filename,
                      qtype=opts.quota_type)

    ctv.run_election(verbose=opts.verbose, debug=opts.debug)
