package com.deepthought.api;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.atLeast;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.testng.Assert.*;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Collections;

import org.springframework.http.ResponseEntity;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import com.deepthought.models.ImageMatrixNode;
import com.deepthought.models.edges.PartOf;
import com.deepthought.models.repository.ImageMatrixNodeRepository;
import com.deepthought.models.repository.PartOfRepository;
import com.deepthought.api.dto.ImageIngestRequest;
import com.deepthought.image.ImageProcessingService;

/**
 * Unit tests for ImageIngestionController. Tests request validation,
 * error handling, and full flow including object crops.
 */
@Test(groups = "Regression")
public class ImageIngestionControllerTests {

	private ImageIngestionController controller;
	private ImageProcessingService image_processing_service;
	private ImageMatrixNodeRepository image_matrix_repo;
	private PartOfRepository part_of_repo;

	private static final String TINY_BASE64_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

	@BeforeMethod
	public void setUp() throws Exception {
		image_processing_service = mock(ImageProcessingService.class);
		image_matrix_repo = mock(ImageMatrixNodeRepository.class);
		part_of_repo = mock(PartOfRepository.class);

		controller = new ImageIngestionController();
		setField(controller, "image_processing_service", image_processing_service);
		setField(controller, "image_matrix_repo", image_matrix_repo);
		setField(controller, "part_of_repo", part_of_repo);

		int[][][] rgb = new int[1][1][3];
		rgb[0][0][0] = 255;
		rgb[0][0][1] = 0;
		rgb[0][0][2] = 0;
		when(image_processing_service.decodeToRgbMatrix(any())).thenReturn(rgb);
		when(image_processing_service.computeOutline(any())).thenReturn(rgb);
		when(image_processing_service.computePca(any())).thenReturn(rgb);
		when(image_processing_service.computeBlackAndWhite(any())).thenReturn(rgb);
		when(image_processing_service.detectAndCropObjects(any())).thenReturn(Collections.emptyList());

		when(image_matrix_repo.save(any(ImageMatrixNode.class))).thenAnswer(inv -> {
			ImageMatrixNode n = inv.getArgument(0);
			if (n.getId() == null) {
				n.setId(1L);
			}
			return n;
		});
	}

	private void setField(Object target, String name, Object value) throws Exception {
		Field f = ImageIngestionController.class.getDeclaredField(name);
		f.setAccessible(true);
		f.set(target, value);
	}

	@Test
	public void ingest_returns200_withValidBase64Image() {
		ImageIngestRequest request = new ImageIngestRequest(TINY_BASE64_PNG);
		ResponseEntity<?> response = controller.ingest(request);
		assertNotNull(response);
		assertEquals(response.getStatusCodeValue(), 200);
		assertTrue(response.getBody() instanceof ImageMatrixNode);
		ImageMatrixNode original = (ImageMatrixNode) response.getBody();
		assertEquals(original.getType(), "ORIGINAL");
		assertEquals(original.getWidth(), 1);
		assertEquals(original.getHeight(), 1);
	}

	@Test
	public void ingest_createsPartOfRelationshipsForDerivedNodes() {
		controller.ingest(new ImageIngestRequest(TINY_BASE64_PNG));
		verify(part_of_repo, atLeast(3)).save(any(PartOf.class));
	}

	@Test
	public void ingest_createsCroppedObjectNodes_whenObjectsDetected() {
		int[][][] crop1 = new int[10][10][3];
		int[][][] crop2 = new int[15][20][3];
		when(image_processing_service.detectAndCropObjects(any()))
				.thenReturn(Arrays.asList(crop1, crop2));
		ResponseEntity<?> response = controller.ingest(new ImageIngestRequest(TINY_BASE64_PNG));
		assertEquals(response.getStatusCodeValue(), 200);
		verify(image_matrix_repo, times(6)).save(any(ImageMatrixNode.class));
		verify(part_of_repo, times(5)).save(any(PartOf.class));
	}

	@Test
	public void ingest_returns400_whenRequestNull() {
		ResponseEntity<?> response = controller.ingest(null);
		assertNotNull(response);
		assertEquals(response.getStatusCodeValue(), 400);
	}

	@Test
	public void ingest_returns400_whenImageMissing() {
		ImageIngestRequest request = new ImageIngestRequest();
		request.setImage(null);
		ResponseEntity<?> response = controller.ingest(request);
		assertEquals(response.getStatusCodeValue(), 400);
	}

	@Test
	public void ingest_returns400_whenImageEmpty() {
		ImageIngestRequest request = new ImageIngestRequest("");
		ResponseEntity<?> response = controller.ingest(request);
		assertEquals(response.getStatusCodeValue(), 400);
	}

	@Test
	public void ingest_returns400_whenImageWhitespaceOnly() {
		ImageIngestRequest request = new ImageIngestRequest("   \t\n  ");
		ResponseEntity<?> response = controller.ingest(request);
		assertEquals(response.getStatusCodeValue(), 400);
	}

	@Test
	public void ingest_returns400_whenDecodeThrowsIllegalArgumentException() throws IOException {
		when(image_processing_service.decodeToRgbMatrix(any()))
				.thenThrow(new IllegalArgumentException("Invalid base64"));
		ImageIngestRequest request = new ImageIngestRequest("invalid");
		ResponseEntity<?> response = controller.ingest(request);
		assertEquals(response.getStatusCodeValue(), 400);
		assertNotNull(response.getBody());
		assertTrue(response.getBody().toString().contains("Invalid base64"));
	}

	@Test
	public void ingest_returns500_whenDecodeThrowsIOException() throws IOException {
		when(image_processing_service.decodeToRgbMatrix(any()))
				.thenThrow(new IOException("Corrupt image data"));
		ImageIngestRequest request = new ImageIngestRequest("invalid");
		ResponseEntity<?> response = controller.ingest(request);
		assertEquals(response.getStatusCodeValue(), 500);
		assertNotNull(response.getBody());
		assertTrue(response.getBody().toString().contains("Failed to decode image"));
	}

	@Test
	public void ingest_returns500_whenProcessingThrowsGenericException() throws IOException {
		when(image_processing_service.computeOutline(any()))
				.thenThrow(new RuntimeException("OpenCV error"));
		ImageIngestRequest request = new ImageIngestRequest(TINY_BASE64_PNG);
		ResponseEntity<?> response = controller.ingest(request);
		assertEquals(response.getStatusCodeValue(), 500);
		assertNotNull(response.getBody());
		assertTrue(response.getBody().toString().contains("Processing error"));
	}
}
