package com.qanairy.image;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

import javax.annotation.PostConstruct;
import javax.imageio.ImageIO;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import nu.pattern.OpenCV;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * Service for decoding images and producing derived RGB matrices: outline
 * (Canny), PCA transformation, black-and-white, and contour-based object crops.
 */
@Service
public class ImageProcessingService {

	private static final Logger log = LoggerFactory.getLogger(ImageProcessingService.class);

	/** Max dimension (width or height) to stay within Neo4j property size limits. */
	private static final int MAX_DIMENSION = 512;

	/** Call once to load OpenCV native libs (e.g. in tests). */
	public static void loadOpenCV() {
		OpenCV.loadShared();
	}

	@PostConstruct
	public void init() {
		try {
			loadOpenCV();
		} catch (Throwable t) {
			log.warn("OpenCV native libraries unavailable on this platform; image endpoints will fail at request time: {}", t.getMessage());
		}
	}

	/**
	 * Decodes a base64-encoded image string into an RGB matrix (int[height][width][3]).
	 * Downscales if either dimension exceeds MAX_DIMENSION.
	 */
	public int[][][] decodeToRgbMatrix(String base64_image) throws IOException {
		byte[] bytes = Base64.getDecoder().decode(base64_image);
		if (bytes == null || bytes.length == 0) {
			throw new IllegalArgumentException("Invalid or empty base64 image data");
		}
		BufferedImage img = ImageIO.read(new ByteArrayInputStream(bytes));
		if (img == null) {
			throw new IllegalArgumentException("Could not decode image; unsupported format");
		}
		int[][][] rgb = bufferedImageToRgbMatrix(img);
		return downscaleIfNeeded(rgb);
	}

	/**
	 * Produces an outline (edge) RGB matrix using Canny edge detection.
	 */
	public int[][][] computeOutline(int[][][] rgb) {
		Mat mat = rgbMatrixToMat(rgb);
		Mat gray = new Mat();
		Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);
		Mat blurred = new Mat();
		Imgproc.GaussianBlur(gray, blurred, new Size(3, 3), 0);
		Mat edges = new Mat();
		Imgproc.Canny(blurred, edges, 50, 150);
		int[][][] result = matToRgbMatrix(edges, true);
		mat.release();
		gray.release();
		blurred.release();
		edges.release();
		return result;
	}

	/**
	 * Produces a PCA-transformed RGB matrix. Each pixel is treated as a 3D vector;
	 * PCA is applied and the first 3 components are used to reconstruct an RGB-like image.
	 */
	public int[][][] computePca(int[][][] rgb) {
		int h = rgb.length;
		int w = rgb[0].length;
		int n = h * w;
		double[][] data = new double[n][3];
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int idx = y * w + x;
				data[idx][0] = rgb[y][x][0];
				data[idx][1] = rgb[y][x][1];
				data[idx][2] = rgb[y][x][2];
			}
		}
		RealMatrix matrix = MatrixUtils.createRealMatrix(data);
		double[] mean = new double[3];
		for (int c = 0; c < 3; c++) {
			double sum = 0;
			for (int i = 0; i < n; i++) {
				sum += matrix.getEntry(i, c);
			}
			mean[c] = sum / n;
		}
		for (int i = 0; i < n; i++) {
			for (int c = 0; c < 3; c++) {
				matrix.setEntry(i, c, matrix.getEntry(i, c) - mean[c]);
			}
		}
		RealMatrix cov = matrix.transpose().multiply(matrix).scalarMultiply(1.0 / (n - 1));
		EigenDecomposition eig = new EigenDecomposition(cov);
		RealMatrix components = eig.getV();
		RealMatrix transformed = matrix.multiply(components);
		int[][][] result = new int[h][w][3];
		double min0 = Double.MAX_VALUE, max0 = Double.NEGATIVE_INFINITY;
		double min1 = Double.MAX_VALUE, max1 = Double.NEGATIVE_INFINITY;
		double min2 = Double.MAX_VALUE, max2 = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < n; i++) {
			double v0 = transformed.getEntry(i, 0);
			double v1 = transformed.getEntry(i, 1);
			double v2 = transformed.getEntry(i, 2);
			min0 = Math.min(min0, v0); max0 = Math.max(max0, v0);
			min1 = Math.min(min1, v1); max1 = Math.max(max1, v1);
			min2 = Math.min(min2, v2); max2 = Math.max(max2, v2);
		}
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int idx = y * w + x;
				double r0 = transformed.getEntry(idx, 0);
				double r1 = transformed.getEntry(idx, 1);
				double r2 = transformed.getEntry(idx, 2);
				result[y][x][0] = scaleToByte(r0, min0, max0);
				result[y][x][1] = scaleToByte(r1, min1, max1);
				result[y][x][2] = scaleToByte(r2, min2, max2);
			}
		}
		return result;
	}

	/**
	 * Produces a black-and-white (grayscale) RGB matrix where R=G=B.
	 */
	public int[][][] computeBlackAndWhite(int[][][] rgb) {
		int h = rgb.length;
		int w = rgb[0].length;
		int[][][] result = new int[h][w][3];
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int gray = (int) (0.299 * rgb[y][x][0] + 0.587 * rgb[y][x][1] + 0.114 * rgb[y][x][2]);
				gray = Math.max(0, Math.min(255, gray));
				result[y][x][0] = gray;
				result[y][x][1] = gray;
				result[y][x][2] = gray;
			}
		}
		return result;
	}

	/**
	 * Detects objects via contours and returns a list of cropped RGB matrices (one per object).
	 */
	public List<int[][][]> detectAndCropObjects(int[][][] rgb) {
		Mat mat = rgbMatrixToMat(rgb);
		Mat gray = new Mat();
		Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);
		Mat blurred = new Mat();
		Imgproc.GaussianBlur(gray, blurred, new Size(5, 5), 0);
		Mat binary = new Mat();
		Imgproc.threshold(blurred, binary, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
		List<MatOfPoint> contours = new ArrayList<>();
		Mat hierarchy = new Mat();
		Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		int minArea = (mat.rows() * mat.cols()) / 100;
		List<int[][][]> crops = new ArrayList<>();
		for (MatOfPoint contour : contours) {
			double area = Imgproc.contourArea(contour);
			if (area < minArea) {
				continue;
			}
			Rect rect = Imgproc.boundingRect(contour);
			if (rect.width < 10 || rect.height < 10) {
				continue;
			}
			Mat crop = new Mat(mat, rect);
			int[][][] cropRgb = matToRgbMatrix(crop, false);
			crops.add(cropRgb);
			crop.release();
		}
		mat.release();
		gray.release();
		blurred.release();
		binary.release();
		hierarchy.release();
		for (MatOfPoint c : contours) {
			c.release();
		}
		return crops;
	}

	private int scaleToByte(double v, double min, double max) {
		if (max <= min) {
			return 128;
		}
		int scaled = (int) (255 * (v - min) / (max - min));
		return Math.max(0, Math.min(255, scaled));
	}

	private int[][][] bufferedImageToRgbMatrix(BufferedImage img) {
		int w = img.getWidth();
		int h = img.getHeight();
		int[][][] rgb = new int[h][w][3];
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int pixel = img.getRGB(x, y);
				rgb[y][x][0] = (pixel >> 16) & 0xFF;
				rgb[y][x][1] = (pixel >> 8) & 0xFF;
				rgb[y][x][2] = pixel & 0xFF;
			}
		}
		return rgb;
	}

	private Mat rgbMatrixToMat(int[][][] rgb) {
		int h = rgb.length;
		int w = rgb[0].length;
		Mat mat = new Mat(h, w, CvType.CV_8UC3);
		byte[] data = new byte[h * w * 3];
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int idx = (y * w + x) * 3;
				data[idx] = (byte) rgb[y][x][2];
				data[idx + 1] = (byte) rgb[y][x][1];
				data[idx + 2] = (byte) rgb[y][x][0];
			}
		}
		mat.put(0, 0, data);
		return mat;
	}

	private int[][][] matToRgbMatrix(Mat mat, boolean singleChannel) {
		int h = mat.rows();
		int w = mat.cols();
		int channels = mat.channels();
		byte[] data = new byte[h * w * channels];
		mat.get(0, 0, data);
		int[][][] rgb = new int[h][w][3];
		if (singleChannel && channels == 1) {
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					int v = data[y * w + x] & 0xFF;
					rgb[y][x][0] = v;
					rgb[y][x][1] = v;
					rgb[y][x][2] = v;
				}
			}
		} else if (channels >= 3) {
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					int idx = (y * w + x) * 3;
					rgb[y][x][0] = data[idx + 2] & 0xFF;
					rgb[y][x][1] = data[idx + 1] & 0xFF;
					rgb[y][x][2] = data[idx] & 0xFF;
				}
			}
		}
		return rgb;
	}

	private int[][][] downscaleIfNeeded(int[][][] rgb) {
		int h = rgb.length;
		int w = rgb[0].length;
		if (h <= MAX_DIMENSION && w <= MAX_DIMENSION) {
			return rgb;
		}
		double scale = Math.min((double) MAX_DIMENSION / h, (double) MAX_DIMENSION / w);
		int newH = (int) (h * scale);
		int newW = (int) (w * scale);
		int[][][] scaled = new int[newH][newW][3];
		for (int y = 0; y < newH; y++) {
			for (int x = 0; x < newW; x++) {
				int sy = (int) (y / scale);
				int sx = (int) (x / scale);
				sy = Math.min(sy, h - 1);
				sx = Math.min(sx, w - 1);
				scaled[y][x][0] = rgb[sy][sx][0];
				scaled[y][x][1] = rgb[sy][sx][1];
				scaled[y][x][2] = rgb[sy][sx][2];
			}
		}
		log.debug("Downscaled image from {}x{} to {}x{}", w, h, newW, newH);
		return scaled;
	}
}
