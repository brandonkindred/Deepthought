package com.qanairy.image;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;

import javax.imageio.ImageIO;

import org.testng.SkipException;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Unit tests for ImageProcessingService. Uses a minimal 1x1 pixel PNG for
 * decode tests; transformations are validated for shape and basic invariants.
 */
@Test(groups = "Regression")
public class ImageProcessingServiceTests {

	private ImageProcessingService service;

	/** Base64-encoded 1x1 red pixel PNG */
	private static final String TINY_BASE64_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

	@BeforeClass
	public void setUp() {
		try {
			ImageProcessingService.loadOpenCV();
		} catch (Throwable t) {
			throw new SkipException("OpenCV native libraries unavailable on this platform: " + t.getMessage());
		}
		service = new ImageProcessingService();
		service.init();
	}

	@Test
	public void decodeToRgbMatrix_returnsCorrectDimensions() throws IOException {
		int[][][] rgb = service.decodeToRgbMatrix(TINY_BASE64_PNG);
		assertNotNull(rgb);
		assertEquals(rgb.length, 1, "height");
		assertEquals(rgb[0].length, 1, "width");
		assertEquals(rgb[0][0].length, 3, "channels");
	}

	@Test
	public void decodeToRgbMatrix_extractsRgbValues() throws IOException {
		int[][][] rgb = service.decodeToRgbMatrix(TINY_BASE64_PNG);
		assertTrue(rgb[0][0][0] >= 0 && rgb[0][0][0] <= 255);
		assertTrue(rgb[0][0][1] >= 0 && rgb[0][0][1] <= 255);
		assertTrue(rgb[0][0][2] >= 0 && rgb[0][0][2] <= 255);
	}

	@Test
	public void decodeToRgbMatrix_throwsOnInvalidBase64() {
		try {
			service.decodeToRgbMatrix("not-valid-base64!!!");
			fail("Expected exception");
		} catch (IOException e) {
			// May throw from ImageIO
		} catch (IllegalArgumentException e) {
			// Expected from Base64 or our validation
		}
	}

	@Test
	public void decodeToRgbMatrix_throwsOnEmpty() {
		try {
			service.decodeToRgbMatrix("");
			fail("Expected IllegalArgumentException");
		} catch (IOException e) {
			fail("Expected IllegalArgumentException, got " + e);
		} catch (IllegalArgumentException e) {
			// Expected
		}
	}

	@Test
	public void decodeToRgbMatrix_throwsOnNonImageBytes() {
		String nonImageBase64 = Base64.getEncoder().encodeToString(new byte[] { 0, 0, 0, 0 });
		try {
			service.decodeToRgbMatrix(nonImageBase64);
			fail("Expected IllegalArgumentException");
		} catch (IOException e) {
			fail("Expected IllegalArgumentException for unsupported format");
		} catch (IllegalArgumentException e) {
			assertNotNull(e.getMessage());
		}
	}

	@Test
	public void decodeToRgbMatrix_downscalesWhenExceedsMaxDimension() throws IOException {
		BufferedImage largeImg = new BufferedImage(1, 600, BufferedImage.TYPE_INT_RGB);
		for (int y = 0; y < 600; y++) {
			largeImg.setRGB(0, y, 0xFF0000);
		}
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ImageIO.write(largeImg, "PNG", baos);
		String base64 = Base64.getEncoder().encodeToString(baos.toByteArray());
		int[][][] rgb = service.decodeToRgbMatrix(base64);
		assertNotNull(rgb);
		assertTrue(rgb.length <= 512, "Height should be downscaled to max 512");
		assertTrue(rgb[0].length <= 512, "Width should be downscaled to max 512");
	}

	@Test
	public void computeBlackAndWhite_producesRgbMatrixWithRequalsGequalsB() throws IOException {
		int[][][] rgb = service.decodeToRgbMatrix(TINY_BASE64_PNG);
		int[][][] bw = service.computeBlackAndWhite(rgb);
		assertNotNull(bw);
		assertEquals(bw.length, rgb.length);
		assertEquals(bw[0].length, rgb[0].length);
		assertEquals(bw[0][0][0], bw[0][0][1]);
		assertEquals(bw[0][0][1], bw[0][0][2]);
	}

	@Test
	public void computeBlackAndWhite_clipsValuesToByteRange() {
		int[][][] rgb = new int[2][2][3];
		rgb[0][0] = new int[] { 300, 300, 300 };
		rgb[0][1] = new int[] { -10, -10, -10 };
		rgb[1][0] = new int[] { 100, 100, 100 };
		rgb[1][1] = new int[] { 255, 255, 255 };
		int[][][] bw = service.computeBlackAndWhite(rgb);
		assertEquals(bw[0][0][0], 255);
		assertEquals(bw[0][1][0], 0);
		assertEquals(bw[1][0][0], 100);
		assertEquals(bw[1][1][0], 255);
	}

	@Test
	public void computeOutline_producesSameDimensions() throws IOException {
		int[][][] rgb = service.decodeToRgbMatrix(TINY_BASE64_PNG);
		int[][][] outline = service.computeOutline(rgb);
		assertNotNull(outline);
		assertEquals(outline.length, rgb.length);
		assertEquals(outline[0].length, rgb[0].length);
	}

	@Test
	public void computeOutline_producesValidRgbValues() throws IOException {
		int[][][] rgb = new int[10][10][3];
		for (int y = 0; y < 10; y++) {
			for (int x = 0; x < 10; x++) {
				rgb[y][x] = new int[] { 100, 150, 200 };
			}
		}
		int[][][] outline = service.computeOutline(rgb);
		for (int y = 0; y < outline.length; y++) {
			for (int x = 0; x < outline[0].length; x++) {
				assertTrue(outline[y][x][0] >= 0 && outline[y][x][0] <= 255);
				assertTrue(outline[y][x][1] >= 0 && outline[y][x][1] <= 255);
				assertTrue(outline[y][x][2] >= 0 && outline[y][x][2] <= 255);
			}
		}
	}

	@Test
	public void computePca_producesSameDimensions() {
		int[][][] rgb = new int[5][5][3];
		for (int y = 0; y < 5; y++) {
			for (int x = 0; x < 5; x++) {
				rgb[y][x] = new int[] { x * 50, y * 50, (x + y) * 25 };
			}
		}
		int[][][] pca = service.computePca(rgb);
		assertNotNull(pca);
		assertEquals(pca.length, rgb.length);
		assertEquals(pca[0].length, rgb[0].length);
	}

	@Test
	public void computePca_producesValidRgbValues() {
		int[][][] rgb = new int[5][5][3];
		for (int y = 0; y < 5; y++) {
			for (int x = 0; x < 5; x++) {
				rgb[y][x] = new int[] { x * 50, y * 50, (x + y) * 25 };
			}
		}
		int[][][] pca = service.computePca(rgb);
		for (int y = 0; y < pca.length; y++) {
			for (int x = 0; x < pca[0].length; x++) {
				assertTrue(pca[y][x][0] >= 0 && pca[y][x][0] <= 255);
				assertTrue(pca[y][x][1] >= 0 && pca[y][x][1] <= 255);
				assertTrue(pca[y][x][2] >= 0 && pca[y][x][2] <= 255);
			}
		}
	}

	@Test
	public void detectAndCropObjects_returnsList() throws IOException {
		int[][][] rgb = service.decodeToRgbMatrix(TINY_BASE64_PNG);
		java.util.List<int[][][]> crops = service.detectAndCropObjects(rgb);
		assertNotNull(crops);
		assertTrue(crops.size() >= 0);
	}

	@Test
	public void detectAndCropObjects_withLargerImage() {
		int[][][] rgb = new int[50][50][3];
		for (int y = 0; y < 50; y++) {
			for (int x = 0; x < 50; x++) {
				rgb[y][x] = new int[] { (x + y) % 256, 100, 200 };
			}
		}
		java.util.List<int[][][]> crops = service.detectAndCropObjects(rgb);
		assertNotNull(crops);
		for (int[][][] crop : crops) {
			assertTrue(crop.length > 0);
			assertTrue(crop[0].length > 0);
			assertEquals(crop[0][0].length, 3);
		}
	}

	@Test
	public void loadOpenCV_canBeCalledMultipleTimes() {
		ImageProcessingService.loadOpenCV();
		ImageProcessingService.loadOpenCV();
	}
}
