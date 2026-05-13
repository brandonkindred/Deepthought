package com.deepthought.api;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import com.deepthought.models.ImageMatrixNode;
import com.deepthought.models.edges.PartOf;
import com.deepthought.models.repository.ImageMatrixNodeRepository;
import com.deepthought.models.repository.PartOfRepository;
import com.deepthought.api.dto.ImageIngestRequest;
import com.deepthought.image.ImageProcessingService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

/**
 * REST endpoints for image ingestion. Accepts base64-encoded images, creates
 * Neo4j nodes for the original and derived matrices (outline, PCA, B&W, cropped
 * objects), and links them via PART_OF relationships.
 */
@RestController
@RequestMapping("/images")
public class ImageIngestionController {

	private static final Logger log = LoggerFactory.getLogger(ImageIngestionController.class);

	@Autowired
	private ImageProcessingService image_processing_service;

	@Autowired
	private ImageMatrixNodeRepository image_matrix_repo;

	@Autowired
	private PartOfRepository part_of_repo;

	@Operation(summary = "Ingest image and create graph nodes", description = "Accepts a base64-encoded image, creates an original image node, and derived nodes (outline, PCA, black-and-white, cropped objects) with PART_OF relationships to the original.", tags = { "Image Ingestion" })
	@ApiResponses(value = {
			@ApiResponse(responseCode = "200", description = "Successfully ingested image", content = @Content(schema = @Schema(implementation = ImageMatrixNode.class))),
			@ApiResponse(responseCode = "400", description = "Invalid request (missing or invalid base64 image)"),
			@ApiResponse(responseCode = "500", description = "Processing error")
	})
	@RequestMapping(value = "/ingest", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
	public ResponseEntity<?> ingest(@RequestBody ImageIngestRequest request) {
		if (request == null || request.getImage() == null || request.getImage().trim().isEmpty()) {
			return ResponseEntity.badRequest().body("Missing or empty 'image' field in request body");
		}
		try {
			int[][][] rgb = image_processing_service.decodeToRgbMatrix(request.getImage());
			int height = rgb.length;
			int width = rgb[0].length;

			ImageMatrixNode original = new ImageMatrixNode(ImageMatrixNode.Type.ORIGINAL, width, height, rgb);
			original = image_matrix_repo.save(original);

			List<Long> created_ids = new ArrayList<>();
			created_ids.add(original.getId());

			int[][][] outline_rgb = image_processing_service.computeOutline(rgb);
			ImageMatrixNode outline_node = new ImageMatrixNode(ImageMatrixNode.Type.OUTLINE, outline_rgb[0].length, outline_rgb.length, outline_rgb);
			outline_node = image_matrix_repo.save(outline_node);
			part_of_repo.save(new PartOf(outline_node, original));
			created_ids.add(outline_node.getId());

			int[][][] pca_rgb = image_processing_service.computePca(rgb);
			ImageMatrixNode pca_node = new ImageMatrixNode(ImageMatrixNode.Type.PCA, pca_rgb[0].length, pca_rgb.length, pca_rgb);
			pca_node = image_matrix_repo.save(pca_node);
			part_of_repo.save(new PartOf(pca_node, original));
			created_ids.add(pca_node.getId());

			int[][][] bw_rgb = image_processing_service.computeBlackAndWhite(rgb);
			ImageMatrixNode bw_node = new ImageMatrixNode(ImageMatrixNode.Type.BLACK_WHITE, bw_rgb[0].length, bw_rgb.length, bw_rgb);
			bw_node = image_matrix_repo.save(bw_node);
			part_of_repo.save(new PartOf(bw_node, original));
			created_ids.add(bw_node.getId());

			List<int[][][]> object_crops = image_processing_service.detectAndCropObjects(rgb);
			for (int[][][] crop : object_crops) {
				int cw = crop[0].length;
				int ch = crop.length;
				ImageMatrixNode obj_node = new ImageMatrixNode(ImageMatrixNode.Type.CROPPED_OBJECT, cw, ch, crop);
				obj_node = image_matrix_repo.save(obj_node);
				part_of_repo.save(new PartOf(obj_node, original));
				created_ids.add(obj_node.getId());
			}

			log.info("Ingested image {}x{}, created {} nodes", width, height, created_ids.size());
			return ResponseEntity.ok(original);
		} catch (IllegalArgumentException e) {
			log.warn("Invalid image request: {}", e.getMessage());
			return ResponseEntity.badRequest().body(e.getMessage());
		} catch (IOException e) {
			log.error("Failed to decode image: {}", e.getMessage());
			return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Failed to decode image: " + e.getMessage());
		} catch (Exception e) {
			log.error("Image processing error: {}", e.getMessage());
			return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Processing error: " + e.getMessage());
		}
	}
}
