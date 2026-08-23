# Infinite Powder Snow

A small Fabric mod for **Minecraft 26.2** that adds a powder snow version of the vanilla dripstone lava farm.

Powder snow is normally a one-way resource: the only renewable source is snowfall filling a cauldron, which needs a
cold biome and a lot of patience. This mod lets you build the same dripstone contraption players already use for lava
and get an endless supply of powder snow instead.

## How to build the farm

Exactly like a lava farm, with a powder snow block in place of the lava source:

```
  ▓  powder snow          <- the source, never consumed
  ▒  any full block       <- the block the dripstone hangs from
  ▼  pointed dripstone    <- stalactite, tip pointing down
     (air)                <- up to 10 blocks of clear space
  ⌷  cauldron
```

Leave the cauldron under the stalactite tip and it slowly fills, one layer at a time, until it is full. Then right
click it with an empty bucket to take the powder snow out, and it starts filling again. The powder snow block on top
is never used up, so one block feeds the farm forever.

Details that match vanilla dripstone behaviour:

- The stalactite can be several segments long (up to 11), and the cauldron can be up to 11 blocks below the tip.
- Anything solid between the tip and the cauldron blocks the drip, same as for water and lava.
- Filling happens on random ticks at the same rate vanilla uses for lava, so a farm fills at a familiar pace.
- Snowflake particles drip from the tip while the farm is running, and the cauldron plays a sound each time it gains
  a layer.
- As in vanilla, the cauldron has to be completely full (3 layers) before a bucket can pick the powder snow up.

Water and lava farms are untouched and keep working exactly as before.

## How it works

Vanilla's dripstone plumbing is built entirely around `Fluid`s, and powder snow is a block rather than a fluid, so
none of it can be reused directly. The mod runs a parallel copy of that logic through two mixins:

- `PointedDripstoneBlockMixin` hooks `PointedDripstoneBlock.maybeTransferFluid`, which is where vanilla decides
  whether this random tick produces a drip. Hooking there rather than `randomTick` means the mod reuses the random
  value vanilla already rolled, so it does not perturb vanilla's dripping cadence. If the block above the
  stalactite's supporting block is powder snow and there is a fillable cauldron below the tip, it schedules a tick on
  the cauldron just like vanilla does.
- `AbstractCauldronBlockMixin` hooks the cauldron side of that scheduled tick. Vanilla resolves the drip back to a
  `Fluid` there, which powder snow can never be, so the powder snow case is handled first and cancels the vanilla
  body. Everything else falls through untouched.

`PowderSnowDripstone` holds the shared logic. It reimplements the two small private vanilla helpers it needs
(`findBlockVertical` and `canDripThrough`) instead of access-widening them, so the mod has no hard coupling to
private vanilla members.

## Building

Requires JDK 25 (Minecraft 26.2 runs on Java 25).

```sh
./gradlew build
```

The mod jar ends up in `build/libs/`. Drop it into `.minecraft/mods` alongside Fabric Loader 0.19.3 or newer.
Fabric API is not required.

To try it in a dev environment:

```sh
./gradlew runClient   # or runServer
```
