package com.francolepo.infinitepowdersnow.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import com.francolepo.infinitepowdersnow.PowderSnowDripstone;

import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.util.RandomSource;
import net.minecraft.world.level.block.AbstractCauldronBlock;
import net.minecraft.world.level.block.PointedDripstoneBlock;
import net.minecraft.world.level.block.state.BlockState;

/**
 * Receives the scheduled drip on the cauldron side.
 *
 * <p>Vanilla resolves the drip back to a {@code Fluid} here, which powder snow can never be, so the powder snow case
 * is handled first and cancels the vanilla body. Anything that is not a powder snow drip falls through untouched.
 */
@Mixin(AbstractCauldronBlock.class)
public class AbstractCauldronBlockMixin {
	@Inject(
			method = "tick(Lnet/minecraft/world/level/block/state/BlockState;Lnet/minecraft/server/level/ServerLevel;Lnet/minecraft/core/BlockPos;Lnet/minecraft/util/RandomSource;)V",
			at = @At("HEAD"),
			cancellable = true
	)
	private void infinitepowdersnow$receivePowderSnowDrip(BlockState cauldronState, ServerLevel level, BlockPos pos,
			RandomSource random, CallbackInfo ci) {
		if (!PowderSnowDripstone.canCauldronReceivePowderSnow(cauldronState)) {
			return;
		}

		BlockPos tipPos = PointedDripstoneBlock.findStalactiteTipAboveCauldron(level, pos);

		if (tipPos == null || !PowderSnowDripstone.isUnderPowderSnowSource(level, tipPos, level.getBlockState(tipPos))) {
			return;
		}

		PowderSnowDripstone.receivePowderSnowDrip(cauldronState, level, pos);
		ci.cancel();
	}
}
