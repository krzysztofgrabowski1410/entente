# DATC choices

The Diplomacy Adjudicator Test Cases (DATC v3.3, live copy at
[webdiplomacy.net/datc.php](https://webdiplomacy.net/datc.php)) leave a number
of rules questions open -- the "issues" of its section 4 -- and offer
alternative outcomes for them. An adjudicator has to pick one outcome per issue
and keep it. This file lists the picks of the standard ruleset, classic
Diplomacy on its classic map. A bold letter is the DATC's own option letter;
the cases in brackets are the test cases under `tests/datc/cases/` that hold
the choice. The Beleriand rulebook states its own choice for every issue.

## 4.A Convoy issues

- **4.A.1 Multi-route convoy disruption -- b.** A convoy is disrupted only
  when all of its routes are broken. (6.F.9-6.F.12)
- **4.A.2 Convoy paradoxes -- f, the Szykman rule.** A convoyed Move that a
  paradox leaves undecided fails and cuts no support; its fleets count as
  holding and may still be attacked and dislodged. The rule touches only the
  paradox itself: other convoys, even beside it, resolve as usual.
  (6.F.14-6.F.18, 6.F.21-6.F.24, 6.F.26-6.F.28)
- **4.A.3 Convoying to adjacent province -- the written form decides.** A Move
  to a province its army reaches by land is convoyed only when it is written
  `via convoy`, even when the power's own fleets order a matching convoy, and
  a Move so written never falls back to the land route. A Move written without
  `via convoy` to a coastal province its army does not reach is a convoy
  attempt when the fleets on the board could carry the army there, and illegal
  otherwise. A Convoy whose cargo's written Move goes over land is invalid.
  (6.C.6, 6.C.7, 6.E.11, 6.G)
  This departs from the DATC's preferred outcome in six cases:
  - 6.G.1, 6.G.5 and 6.G.6: the two units meet head-to-head over land and
    bounce; they do not swap.
  - 6.G.9: Sweden is dislodged; there is no swap.
  - 6.G.11: there is no paradox; Norway's support is cut and F NWY is
    dislodged.
  - 6.G.14: the dislodger moves over land, so 4.A.7 leaves the dislodged unit
    no effect on its province, and F NWG - NWY succeeds.
- **4.A.4 The origin of the convoyed attack -- a.** An attack from the province
  a support is directed against does not cut that support, even when it
  arrives by convoy. (6.G.13, 6.F.30)
- **4.A.5 Retreat when dislodged by convoy -- b.** A unit may retreat to the
  province its dislodger came from only if the dislodger came by convoy. Both
  the ban and the exception hold for the whole province, whatever the coast.
  (6.H.5, 6.H.10, 6.H.15; 6.H.11, 6.H.12)
- **4.A.6 Convoy path specification -- a.** Orders name no route; any chain of
  convoying fleets counts. (6.C.4)
- **4.A.7 Avoiding a head-to-head battle to bounce a unit -- b.** A unit
  dislodged by a Move over land from the province it was moving to has no
  effect on that province; one dislodged by a convoyed Move from there still
  has. (6.E.1, 6.G.10, 6.G.14, 6.G.15)

## 4.B Coastal issues

- **4.B.1 Omitted coast when two coasts are possible.** The Move is illegal,
  and it is ignored. (6.B.1)
- **4.B.2 Omitted coast when one coast is possible.** The Move goes to the one
  coast it can reach. (6.B.2)
- **4.B.3 Move order to impossible coast -- b.** The Move is illegal; it is not
  read as a Move to the other coast. (6.B.3, 6.D.29)
- **4.B.4 Coast specification in support order.** A support matches a Move by
  its destination province: a support that writes no coast matches any coast,
  and a coast it writes must agree with the Move's. (6.B.4, 6.B.7-6.B.9,
  6.B.15)
- **4.B.5 Wrong coast when it is not required.** The coast is ignored: a unit
  is found by its province, and the order is checked from where the unit really
  stands. (6.B.10, 6.B.11, 6.B.16, 6.B.17)
- **4.B.6 Unknown or irrelevant coasts.** Ignored: a coast written for an army
  is dropped. (6.B.12)
- **4.B.7 Coast specification in build order.** A fleet built in a province
  with two coasts must name one of them, and every build must name a place its
  kind can stand on. (6.B.14, 6.I.2)

## 4.C Unit designation and nationality

- **4.C.1 Missing unit designation.** It cannot happen: the order notation
  requires the unit's kind.
- **4.C.2 Wrong unit designation -- b.** The order is legal: the written kind is
  advisory, and the order binds the unit standing in the province. (6.D.35)
- **4.C.3 Missing unit designation in build order.** It cannot happen, as in
  4.C.1.
- **4.C.4 Building a fleet on an inland province -- a.** The build fails; it is
  never read as an army build. (6.I.2)
- **4.C.5 and 4.C.6 Missing and wrong nationality in support order -- b.** The
  notation writes no nationality, so none can be missing or wrong.

## 4.D Order sets and adjustment orders

- **4.D.1 and 4.D.2 Multiple order sets -- c.** A power's later sheet replaces
  its earlier one; sheets are never merged.
- **4.D.3 Multiple orders to the same unit.** Illegal orders are ignored (see
  4.E.1), and the first remaining order is the unit's order. A Support or a
  Convoy that the board alone forbids is illegal, so the unit's next order
  counts; one that proves invalid only against the other orders -- the
  supported unit received another order, or the cargo's Move goes over land --
  keeps its unit, which holds.
- **4.D.4, 4.D.5 and 4.D.6 Too many build orders, several builds for one
  province, too many disband orders -- b.** Adjustment orders are read top to
  bottom: the first legal ones use up the quota and the later ones fail, an
  illegal order uses none of it, a province takes one build, and repeated
  identical lines count as one order. (6.I.1, 6.I.7, 6.J.1, 6.J.2, 6.B.14)
- **4.D.7 Declining builds -- a.** A power may decline builds with `Waive`; a
  build left unused is declined as well, and each Waive line declines one
  build.
- **4.D.8 Removing a unit in civil disorder -- the 2023 and 2025 rulebook
  wording.** A removal that a power does not order -- in civil disorder, or
  short of the disbands it owes -- takes the unit farthest from the power's
  nearest owned supply centre. Distance counts steps between adjacent
  provinces, the same for armies and fleets; a tie removes a fleet before an
  army, and then goes by the full province name in alphabetical order. A power
  that owns no centre removes everything, and a unit that no owned centre can
  reach counts as farthest. (6.J.2-6.J.12)
- **4.D.9 Receiving hold support in civil disorder -- b.** A foreign
  support-hold of the unit of a power in civil disorder succeeds: a unit
  without an order can be supported to hold. (6.A.12, 6.D.1)

## 4.E Illegal orders and interpretation

- **4.E.1 Illegal orders.** An illegal order is ignored entirely, and a unit
  left with no legal order holds and can be supported to hold. An army's Move
  to a coastal province it does not reach is a convoy attempt when the fleets
  on the board could carry it there -- legal, and the army counts as moving
  even if no fleet convoys it -- and illegal otherwise. (6.A.5, 6.D.8,
  6.D.28-6.D.32, 6.E.14)
- **4.E.2 to 4.E.6 Resolution terminology, implicit, perpetual and proxy
  orders, the "flying Dutchman".** These are questions for a human judge. The
  engine takes one written sheet per power and phase, reads every order on its
  own, infers no order that a sheet does not write, and reports every order as
  succeeding or failing, with the dislodgements beside them.
