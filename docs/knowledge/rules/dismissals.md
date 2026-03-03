# Dismissals (How wickets happen)

This doc summarizes common dismissal types and the typical “who gets credit” intuition.

## Bowled

- The ball hits the stumps and dislodges the bails.
- Credited as a wicket to the **bowler**.

## Caught

- The batter hits the ball and it is caught by a fielder (including the wicketkeeper) before it bounces.
- Credited as a wicket to the **bowler**.

## LBW

- “Leg Before Wicket”: the ball hits the batter’s pad/body and would have hit the stumps (subject to conditions).
- Credited as a wicket to the **bowler**.

See: `rules/lbw.md`

## Stumped

- The wicketkeeper breaks the wicket while the batter is out of their crease, typically off spin or after advancing down the pitch.
- Credited as a wicket to the **bowler**.

## Run out

- A fielder breaks the wicket while the batter(s) are running and the batter is short of their ground.
- Usually **not** credited to the bowler (credited to the fielding side).

## Hit wicket

- The batter dislodges the bails with their bat/body while playing a shot or setting off.
- Credited as a wicket to the **bowler**.

## Retired hurt / retired out

- A batter leaves the field and cannot continue (retired hurt).
- Some cases can be treated as “retired out” depending on rules.
- These are special cases and should be handled explicitly in analysis.

## Practical note for analytics

When people say “wickets” for bowlers, they usually mean **bowler-attributed wickets**, excluding run outs and some rare dismissal types.

This project’s analyst templates follow that convention unless the question explicitly asks for “all dismissals”.

