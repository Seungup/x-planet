//! Per-frame 3D Tiles logic: spawn init tasks, poll messages,
//! traverse the tileset, and render visible tiles.

use crate::tiles3d_native::{
    cesium_resolve_endpoint, fetch_tile_content, fetch_tileset, Tiles3dAuthKind, Tiles3dMessage,
};

use super::NativeApp;

impl NativeApp {
    /// Spawn async init tasks for any uninitialized 3D Tiles layers.
    pub(super) fn tiles3d_spawn_init(&mut self) {
        for ts3d in &mut self.tiles3d_states {
            if !ts3d.is_initialized() && !ts3d.init_spawned {
                ts3d.init_spawned = true;
                let client = ts3d.client.clone();
                let tx = self.tiles3d_tx.clone();
                let layer_name = ts3d.name.clone();

                match &ts3d.auth {
                    Tiles3dAuthKind::CesiumIon {
                        account_token,
                        asset_id,
                    } => {
                        let token = account_token.clone();
                        let aid = *asset_id;
                        self.rt.spawn(async move {
                            let result =
                                match cesium_resolve_endpoint(&client, &token, aid).await {
                                    Ok((endpoint_url, access_token)) => {
                                        match fetch_tileset(
                                            &client,
                                            &endpoint_url,
                                            Some(&access_token),
                                        )
                                        .await
                                        {
                                            Ok((tileset, base_url)) => Ok((
                                                tileset,
                                                base_url,
                                                Some(access_token),
                                            )),
                                            Err(e) => Err(e),
                                        }
                                    }
                                    Err(e) => Err(e),
                                };
                            let _ = tx.send(Tiles3dMessage::Initialized {
                                layer_name,
                                result: Box::new(result),
                            });
                        });
                    }
                    Tiles3dAuthKind::Google { api_key } => {
                        let key = api_key.clone();
                        self.rt.spawn(async move {
                            let url = format!(
                                "https://tile.googleapis.com/v1/3dtiles/root.json?key={}",
                                key
                            );
                            let result = match fetch_tileset(&client, &url, None).await {
                                Ok((tileset, base_url)) => {
                                    Ok((tileset, base_url, None))
                                }
                                Err(e) => Err(e),
                            };
                            let _ = tx.send(Tiles3dMessage::Initialized {
                                layer_name,
                                result: Box::new(result),
                            });
                        });
                    }
                }
            }
        }
    }

    /// Poll the 3D Tiles message channel and process init/content results.
    pub(super) fn tiles3d_poll_messages(&mut self) {
        while let Ok(msg) = self.tiles3d_rx.try_recv() {
            match msg {
                Tiles3dMessage::Initialized {
                    layer_name,
                    result,
                } => {
                    if let Some(ts3d) = self
                        .tiles3d_states
                        .iter_mut()
                        .find(|s| s.name == layer_name)
                    {
                        match *result {
                            Ok((tileset, base_url, access_token)) => {
                                let tile_count =
                                    x_planets_tiles::tiles3d::tileset::tile_count(
                                        &tileset.root,
                                    );
                                log::info!(
                                    "[{}] 3D Tiles initialized: {} tiles",
                                    layer_name,
                                    tile_count,
                                );
                                ts3d.tileset = Some(tileset);
                                ts3d.base_url = base_url;
                                ts3d.access_token = access_token;
                            }
                            Err(e) => {
                                log::error!(
                                    "[{}] 3D Tiles init failed: {}",
                                    layer_name,
                                    e
                                );
                            }
                        }
                    }
                }
                Tiles3dMessage::ContentLoaded {
                    layer_name,
                    content_uri,
                    result,
                    generation,
                } => {
                    if let Some(ts3d) = self
                        .tiles3d_states
                        .iter_mut()
                        .find(|s| s.name == layer_name)
                    {
                        ts3d.pending_uris.remove(&content_uri);

                        if generation < ts3d.generation.saturating_sub(2) {
                            ts3d.stale_loads_skipped += 1;
                            log::debug!(
                                "[{}] skipping stale load: {} (gen {} vs current {})",
                                layer_name, content_uri, generation, ts3d.generation,
                            );
                            continue;
                        }

                        match result {
                            Ok(decoded) => {
                                log::debug!(
                                    "[{}] 3D tile loaded: {} ({} meshes)",
                                    layer_name,
                                    content_uri,
                                    decoded.meshes.len(),
                                );
                                let gpu = self.gpu.as_ref().unwrap();
                                let renderer =
                                    self.model3d_renderer.as_ref().unwrap();
                                ts3d.upload_decoded_tile(
                                    gpu,
                                    renderer,
                                    &content_uri,
                                    &decoded,
                                );
                            }
                            Err(e) => {
                                log::warn!(
                                    "[{}] 3D tile failed: {} - {}",
                                    layer_name,
                                    content_uri,
                                    e
                                );
                                // Mark as loaded to prevent infinite re-fetch.
                                ts3d.loaded_uris.insert(content_uri.clone());
                            }
                        }
                    }
                }
                Tiles3dMessage::ExternalTilesetLoaded {
                    layer_name,
                    content_uri,
                    tileset,
                    base_url,
                    generation,
                } => {
                    if let Some(ts3d) = self
                        .tiles3d_states
                        .iter_mut()
                        .find(|s| s.name == layer_name)
                    {
                        ts3d.pending_uris.remove(&content_uri);

                        if generation < ts3d.generation.saturating_sub(2) {
                            ts3d.stale_loads_skipped += 1;
                            continue;
                        }

                        if let Some(main_tileset) = &mut ts3d.tileset {
                            let spliced =
                                x_planets_tiles::tiles3d::tileset::splice_external_tileset(
                                    &mut main_tileset.root,
                                    &ts3d.base_url,
                                    &content_uri,
                                    &tileset,
                                    &base_url,
                                );
                            if spliced {
                                ts3d.loaded_uris.insert(content_uri.clone());
                                let new_count =
                                    x_planets_tiles::tiles3d::tileset::tile_count(
                                        &main_tileset.root,
                                    );
                                log::info!(
                                    "[{}] spliced external tileset from {} (tree now {} tiles)",
                                    layer_name, content_uri, new_count,
                                );
                            } else {
                                log::warn!(
                                    "[{}] failed to splice external tileset: {} not found in tree",
                                    layer_name, content_uri,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Traverse initialized 3D Tiles layers, spawn loads, and render.
    pub(super) fn tiles3d_traverse_and_render(&mut self, view: &wgpu::TextureView) {
        let engine = &self.controller.as_ref().unwrap().engine;
        let show_stats = std::env::var("XPLANETS_STATS").as_deref() == Ok("1");

        for ts3d in &mut self.tiles3d_states {
            if !ts3d.is_initialized() {
                continue;
            }

            // Increment generation each traversal for stale request detection.
            ts3d.generation += 1;
            let current_generation = ts3d.generation;

            let tileset = ts3d.tileset.as_ref().unwrap();
            let camera =
                x_planets_core::tiles3d_pipeline::viewport_to_traversal_camera(
                    &engine.viewport,
                );
            let config =
                x_planets_tiles::tiles3d::traversal::TraversalConfig {
                    max_sse: ts3d.max_sse,
                    tile_budget: ts3d.tile_budget,
                    screen_height: engine.viewport.height as f64,
                    fov_y: x_planets_core::tiles3d_pipeline::traversal_fov_y(),
                };

            let traversal =
                x_planets_tiles::tiles3d::traversal::traverse_tileset(
                    tileset,
                    &ts3d.base_url,
                    &camera,
                    &ts3d.loaded_uris,
                    &config,
                );

            // Spawn loads for missing tiles (tagged with current generation).
            for req in &traversal.load_requests {
                if ts3d.pending_uris.contains(&req.content_uri) {
                    continue;
                }
                if ts3d.pending_uris.len() >= ts3d.max_concurrent {
                    break;
                }
                ts3d.pending_uris.insert(req.content_uri.clone());

                let client = ts3d.client.clone();
                let tx = self.tiles3d_tx.clone();
                let layer_name = ts3d.name.clone();
                let content_uri = req.content_uri.clone();
                let access_token = ts3d.access_token.clone();

                self.rt.spawn(async move {
                    let fetch_result = fetch_tile_content(
                        &client,
                        &content_uri,
                        access_token.as_deref(),
                    )
                    .await;
                    match fetch_result {
                        Ok(bytes) => {
                            match x_planets_tiles::tiles3d::decoder::decode_3d_tile(
                                &bytes,
                                &content_uri,
                            ) {
                                Ok(x_planets_tiles::tiles3d::decoder::Tiles3dContent::Mesh(decoded)) => {
                                    let _ = tx.send(Tiles3dMessage::ContentLoaded {
                                        layer_name,
                                        content_uri,
                                        result: Ok(decoded),
                                        generation: current_generation,
                                    });
                                }
                                Ok(x_planets_tiles::tiles3d::decoder::Tiles3dContent::ExternalTileset {
                                    tileset, base_url, content_uri,
                                }) => {
                                    let _ = tx.send(Tiles3dMessage::ExternalTilesetLoaded {
                                        layer_name,
                                        content_uri,
                                        tileset,
                                        base_url,
                                        generation: current_generation,
                                    });
                                }
                                Err(e) => {
                                    let _ = tx.send(Tiles3dMessage::ContentLoaded {
                                        layer_name,
                                        content_uri,
                                        result: Err(e.to_string()),
                                        generation: current_generation,
                                    });
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Tiles3dMessage::ContentLoaded {
                                layer_name,
                                content_uri,
                                result: Err(e),
                                generation: current_generation,
                            });
                        }
                    }
                });
            }

            // Unload tiles no longer needed (track GPU memory).
            for uri in &traversal.unload_set {
                if let Some(evicted) = ts3d.gpu_tiles.remove(uri) {
                    ts3d.total_gpu_bytes = ts3d.total_gpu_bytes.saturating_sub(evicted.gpu_bytes);
                }
                ts3d.loaded_uris.remove(uri);
            }

            // Update transforms and render.
            if !traversal.render_set.is_empty() {
                let gpu = self.gpu.as_ref().unwrap();
                let (uniforms, camera_ecef) =
                    x_planets_core::tiles3d_pipeline::build_tiles3d_uniforms(
                        &engine.viewport,
                    );

                ts3d.update_render_transforms(
                    &gpu.queue,
                    &traversal.render_set,
                    camera_ecef,
                    1.0, // opacity
                );

                let models = ts3d.collect_render_models(&traversal.render_set);
                if let Some(model3d_renderer) = &self.model3d_renderer {
                    model3d_renderer.render_models_with_uniforms(
                        gpu,
                        view,
                        &uniforms,
                        &models,
                    );
                }
            }

            // Log stats if XPLANETS_STATS=1 is set.
            if show_stats && ts3d.generation % 60 == 0 {
                let mut stats = ts3d.stats();
                stats.rendered_tiles = traversal.render_set.len();
                log::info!("[{}] {}", ts3d.name, stats);
            }
        }
    }
}
