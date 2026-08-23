package com.francolepo.infinitepowdersnow;

import java.util.Optional;
import java.util.function.BiPredicate;
import java.util.function.Predicate;

import org.jspecify.annotations.Nullable;

import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.particles.ParticleTypes;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.sounds.SoundEvents;
import net.minecraft.sounds.SoundSource;
import net.minecraft.tags.BlockTags;
import net.minecraft.world.level.BlockGetter;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.LevelAccessor;
import net.minecraft.world.level.LevelReader;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.LayeredCauldronBlock;
import net.minecraft.world.level.block.SpeleothemBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.SpeleothemThickness;
import net.minecraft.world.level.gameevent.GameEvent;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.shapes.BooleanOp;
import net.minecraft.world.phys.shapes.Shapes;
import net.minecraft.world.phys.shapes.VoxelShape;

/**
 * A powder snow copy of the vanilla dripstone fluid farm.
 *
 * <p>Vanilla only ever drips {@link net.minecraft.world.level.material.Fluid}s, and powder snow is a block rather
 * than a fluid, so none of the vanilla plumbing can be reused directly. This class mirrors the shape of
 * {@code PointedDripstoneBlock}: a stalactite whose supporting block has powder snow sitting on top of it drips into
 * the first cauldron below the stalactite tip, without ever consuming the powder snow source.
 *
 * <p>The few private vanilla helpers this needs ({@code findBlockVertical} and {@code canDripThrough}) are
 * reimplemented here rather than access-widened, so the mod keeps working as long as the public block state
 * properties stay the same.
 */
public final class PowderSnowDripstone {
	/**
	 * Chance per random tick that a stalactite under a powder snow source drips. Deliberately the same rate vanilla
	 * uses for lava ({@code LAVA_TRANSFER_PROBABILITY_PER_RANDOM_TICK}), so a powder snow farm fills at the same
	 * "slow" pace players already know from a lava farm.
	 */
	public static final float POWDER_SNOW_TRANSFER_PROBABILITY_PER_RANDOM_TICK = 0.05859375F;

	/** Chance per client animation tick that an ambient drip particle spawns under the stalactite tip. */
	public static final float DRIP_PROBABILITY_PER_ANIMATE_TICK = 0.12F;

	private static final int MAX_SEARCH_LENGTH_WHEN_CHECKING_DRIP_TYPE = 11;
	private static final int MAX_SEARCH_LENGTH_BETWEEN_STALACTITE_TIP_AND_CAULDRON = 11;

	/** Mirrors {@code PointedDripstoneBlock.STALACTITE_DRIP_START_PIXEL}, the bottom of the downwards tip shape. */
	private static final double STALACTITE_DRIP_START_PIXEL = 5.0 / 16.0;
	private static final double PIXEL_SIZE = 0.0625;

	/** Mirrors {@code PointedDripstoneBlock.REQUIRED_SPACE_TO_DRIP_THROUGH_NON_SOLID_BLOCK}. */
	private static final VoxelShape REQUIRED_SPACE_TO_DRIP_THROUGH_NON_SOLID_BLOCK = Block.column(4.0, 0.0, 16.0);

	private PowderSnowDripstone() {
	}

	/**
	 * Server side half of the farm, called for every random tick of a pointed dripstone with the same random value
	 * vanilla uses for its own water/lava transfer, so this does not perturb vanilla's dripping cadence.
	 */
	public static void maybeTransferPowderSnow(BlockState state, ServerLevel level, BlockPos pos, float randomValue) {
		if (randomValue >= POWDER_SNOW_TRANSFER_PROBABILITY_PER_RANDOM_TICK) {
			return;
		}

		if (!isStalactiteStartPos(state, level, pos) || !isUnderPowderSnowSource(level, pos, state)) {
			return;
		}

		BlockPos tipPos = findTip(state, level, pos, MAX_SEARCH_LENGTH_WHEN_CHECKING_DRIP_TYPE);

		if (tipPos == null) {
			return;
		}

		BlockPos cauldronPos = findFillableCauldronBelowStalactiteTip(level, tipPos);

		if (cauldronPos == null) {
			return;
		}

		spawnFallingDripParticle(level, tipPos, level.getBlockState(tipPos));

		int fallDistance = tipPos.getY() - cauldronPos.getY();
		int delay = 50 + fallDistance;
		level.scheduleTick(cauldronPos, level.getBlockState(cauldronPos).getBlock(), delay);
	}

	/** Whether the block resting on top of this stalactite's supporting block is a powder snow source. */
	public static boolean isUnderPowderSnowSource(Level level, BlockPos stalactitePos, BlockState stalactiteState) {
		if (!isStalactite(stalactiteState)) {
			return false;
		}

		return findRootBlock(level, stalactitePos, stalactiteState, MAX_SEARCH_LENGTH_WHEN_CHECKING_DRIP_TYPE)
				.map(rootPos -> level.getBlockState(rootPos.above()).is(Blocks.POWDER_SNOW))
				.orElse(false);
	}

	/** Whether this cauldron still has room for another layer of powder snow. */
	public static boolean canCauldronReceivePowderSnow(BlockState cauldronState) {
		if (cauldronState.is(Blocks.CAULDRON)) {
			return true;
		}

		return cauldronState.is(Blocks.POWDER_SNOW_CAULDRON)
				&& cauldronState.getValue(LayeredCauldronBlock.LEVEL) < LayeredCauldronBlock.MAX_FILL_LEVEL;
	}

	/**
	 * Adds one layer of powder snow to the cauldron. The source block above the stalactite is intentionally left
	 * alone, which is what makes the farm infinite.
	 */
	public static void receivePowderSnowDrip(BlockState cauldronState, ServerLevel level, BlockPos pos) {
		BlockState newState;

		if (cauldronState.is(Blocks.CAULDRON)) {
			newState = Blocks.POWDER_SNOW_CAULDRON.defaultBlockState();
		} else if (canCauldronReceivePowderSnow(cauldronState)) {
			newState = cauldronState.setValue(LayeredCauldronBlock.LEVEL, cauldronState.getValue(LayeredCauldronBlock.LEVEL) + 1);
		} else {
			return;
		}

		level.setBlockAndUpdate(pos, newState);
		level.gameEvent(GameEvent.BLOCK_CHANGE, pos, GameEvent.Context.of(newState));
		level.playSound(null, pos, SoundEvents.BUCKET_EMPTY_POWDER_SNOW, SoundSource.BLOCKS, 1.0F, 1.0F);
	}

	/** Client side ambient particle under a stalactite tip that is hanging under powder snow. */
	public static void spawnDripParticle(Level level, BlockPos tipPos, BlockState tipState) {
		Vec3 dripPos = getDripPos(tipPos, tipState);
		level.addParticle(ParticleTypes.SNOWFLAKE, dripPos.x, dripPos.y, dripPos.z, 0.0, 0.0, 0.0);
	}

	private static void spawnFallingDripParticle(ServerLevel level, BlockPos tipPos, BlockState tipState) {
		Vec3 dripPos = getDripPos(tipPos, tipState);
		level.sendParticles(ParticleTypes.SNOWFLAKE, dripPos.x, dripPos.y, dripPos.z, 1, 0.0, 0.0, 0.0, 0.0);
	}

	private static Vec3 getDripPos(BlockPos tipPos, BlockState tipState) {
		Vec3 offset = tipState.getOffset(tipPos);
		return new Vec3(
				tipPos.getX() + 0.5 + offset.x,
				tipPos.getY() + STALACTITE_DRIP_START_PIXEL - PIXEL_SIZE,
				tipPos.getZ() + 0.5 + offset.z
		);
	}

	@Nullable
	private static BlockPos findFillableCauldronBelowStalactiteTip(Level level, BlockPos tipPos) {
		Predicate<BlockState> cauldronPredicate = PowderSnowDripstone::canCauldronReceivePowderSnow;
		BiPredicate<BlockPos, BlockState> pathPredicate = (pos, state) -> canDripThrough(level, pos, state);
		return findBlockVertical(level, tipPos, Direction.DOWN, pathPredicate, cauldronPredicate,
				MAX_SEARCH_LENGTH_BETWEEN_STALACTITE_TIP_AND_CAULDRON).orElse(null);
	}

	/** The first non-speleothem block above the stalactite, i.e. the block the stalactite hangs from. */
	private static Optional<BlockPos> findRootBlock(Level level, BlockPos pos, BlockState stalactiteState, int maxSearchLength) {
		Direction tipDirection = stalactiteState.getValue(SpeleothemBlock.TIP_DIRECTION);
		BiPredicate<BlockPos, BlockState> pathPredicate = (pathPos, state) -> state.is(stalactiteState.getBlock())
				&& state.getValue(SpeleothemBlock.TIP_DIRECTION) == tipDirection;
		return findBlockVertical(level, pos, tipDirection.getOpposite(), pathPredicate,
				state -> !state.is(stalactiteState.getBlock()), maxSearchLength);
	}

	@Nullable
	private static BlockPos findTip(BlockState stalactiteState, LevelAccessor level, BlockPos pos, int maxSearchLength) {
		if (isTip(stalactiteState)) {
			return pos;
		}

		Direction searchDirection = stalactiteState.getValue(SpeleothemBlock.TIP_DIRECTION);
		BiPredicate<BlockPos, BlockState> pathPredicate = (pathPos, state) -> state.is(stalactiteState.getBlock())
				&& state.getValue(SpeleothemBlock.TIP_DIRECTION) == searchDirection;
		return findBlockVertical(level, pos, searchDirection, pathPredicate, PowderSnowDripstone::isTip, maxSearchLength)
				.orElse(null);
	}

	/** Reimplementation of the private {@code SpeleothemBlock.findBlockVertical}. */
	private static Optional<BlockPos> findBlockVertical(LevelAccessor level, BlockPos pos, Direction direction,
			BiPredicate<BlockPos, BlockState> pathPredicate, Predicate<BlockState> targetPredicate, int maxSteps) {
		BlockPos.MutableBlockPos mutablePos = pos.mutable();

		for (int i = 1; i < maxSteps; i++) {
			mutablePos.move(direction);
			BlockState state = level.getBlockState(mutablePos);

			if (targetPredicate.test(state)) {
				return Optional.of(mutablePos.immutable());
			}

			if (level.isOutsideBuildHeight(mutablePos.getY()) || !pathPredicate.test(mutablePos, state)) {
				return Optional.empty();
			}
		}

		return Optional.empty();
	}

	/** Reimplementation of the private {@code PointedDripstoneBlock.canDripThrough}. */
	private static boolean canDripThrough(BlockGetter level, BlockPos pos, BlockState state) {
		if (state.isAir()) {
			return true;
		} else if (state.isSolidRender()) {
			return false;
		} else if (!state.getFluidState().isEmpty()) {
			return false;
		} else {
			VoxelShape collisionShape = state.getCollisionShape(level, pos);
			return !Shapes.joinIsNotEmpty(REQUIRED_SPACE_TO_DRIP_THROUGH_NON_SOLID_BLOCK, collisionShape, BooleanOp.AND);
		}
	}

	private static boolean isStalactite(BlockState state) {
		return state.is(BlockTags.SPELEOTHEMS) && state.getValue(SpeleothemBlock.TIP_DIRECTION) == Direction.DOWN;
	}

	private static boolean isStalactiteStartPos(BlockState state, LevelReader level, BlockPos pos) {
		return isStalactite(state) && !level.getBlockState(pos.above()).is(state.getBlock());
	}

	private static boolean isTip(BlockState state) {
		return state.is(BlockTags.SPELEOTHEMS) && state.getValue(SpeleothemBlock.THICKNESS) == SpeleothemThickness.TIP;
	}

	public static boolean isFreeHangingStalactite(BlockState state) {
		return isStalactite(state)
				&& state.getValue(SpeleothemBlock.THICKNESS) == SpeleothemThickness.TIP
				&& !state.getValue(SpeleothemBlock.WATERLOGGED);
	}
}
