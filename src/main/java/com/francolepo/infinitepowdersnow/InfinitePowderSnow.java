package com.francolepo.infinitepowdersnow;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.fabricmc.api.ModInitializer;

public class InfinitePowderSnow implements ModInitializer {
	public static final String MOD_ID = "infinitepowdersnow";
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	@Override
	public void onInitialize() {
		LOGGER.info("Powder snow dripstone farms enabled");
	}
}
