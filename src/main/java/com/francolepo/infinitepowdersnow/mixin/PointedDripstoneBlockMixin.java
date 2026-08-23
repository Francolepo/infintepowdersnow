package com.francolepo.infinitepowdersnow.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import com.francolepo.infinitepowdersnow.PowderSnowDripstone;

import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.PointedDripstoneBlock;
import net.minecraft.world.level.block.state.BlockState;

/**
 * Runs the powder snow drip check alongside vanilla's water/lava one.
 *
 * <p>Hooking {@code maybeTransferFluid} rather than {@code randomTick} means the same random value vanilla rolled for
 * this tick is reused, so no extra number is pulled from the block's {@code RandomSource}. Vanilla's own body then
 * runs untouched: powder snow is not a fluid, so it bails out on its own.
 */
@Mixin(PointedDripstoneBlock.class)
public class PointedDripstoneBlockMixin {
	@Inject(method = "maybeTransferFluid", at = @At("HEAD"))
	private static void infinitepowdersnow$maybeTransferPowderSnow(BlockState state, ServerLevel level, BlockPos pos,
			float randomValue, CallbackInfo ci) {
		PowderSnowDripstone.maybeTransferPowderSnow(state, level, pos, randomValue);
	}
}
