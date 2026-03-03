# Common T20 metrics (dots, boundaries, rates)

This doc defines several commonly used limited-overs metrics and notes common definition differences.

## Dot ball

A dot ball is a delivery that results in **0 runs**.

Two common definitions:

- **Dot (total runs)**: `runs_total == 0`
- **Dot (off the bat)**: `runs_batter == 0` (byes/leg-byes count as “not off the bat”)

Be explicit about which definition you use.

## Boundary

A boundary is usually:
- **4** or **6** off the bat (`runs_batter in {4,6}`)

Some analyses treat byes/leg-byes boundaries separately.

## Run rate (batting team)

> Run rate = runs per over

Use legal balls to compute overs:
- overs = legal_balls / 6
- run_rate = runs_total / overs

## Extras rate

> Extras rate = extras per over

Useful to evaluate discipline (wides/no-balls) vs opposition pressure.

## Wickets (bowler-attributed)

For many bowling leaderboards, “wickets” excludes:
- run outs
- some rare dismissal types not credited to the bowler

If you include all dismissals, say so explicitly.

