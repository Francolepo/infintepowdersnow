package com.francolepo.infinitepowdersnow.client.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import com.francolepo.infinitepowdersnow.PowderSnowDripstone;

import net.minecraft.core.BlockPos;
import net.minecraft.util.RandomSource;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.PointedDripstoneBlock;
import net.minecraft.world.level.block.state.BlockState;

/** Ambient snowflakes under a stalactite tip that is hanging beneath powder snow, matching vanilla's drip particles. */
@Mixin(PointedDripstoneBlock.class)
public class PointedDripstoneBlockClientMixin {
	@Inject(method = "animateTick", at = @At("HEAD"))
	private void infinitepowdersnow$animatePowderSnowDrip(BlockState state, Level level, BlockPos pos,
			RandomSource random, CallbackInfo ci) {
		if (!PowderSnowDripstone.isFreeHangingStalactite(state)) {
			return;
		}

		if (random.nextFloat() >= PowderSnowDripstone.DRIP_PROBABILITY_PER_ANIMATE_TICK) {
			return;
		}

		if (PowderSnowDripstone.isUnderPowderSnowSource(level, pos, state)) {
			PowderSnowDripstone.spawnDripParticle(level, pos, state);
		}
	}
}
