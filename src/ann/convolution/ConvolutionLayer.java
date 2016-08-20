package ann.convolution;

import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.util.*;

import ann.*;

/**
 * input layer must always be same size (so scale it)
 * @author alexanderbraylan
 *
 */
public class ConvolutionLayer extends Layer<ConvolutionNode> {

	private final BiasNode biasNode = BiasNode.create();
	private ConvolutionNode[][] nodeMap;
	private Dimension dimensions;
	private Set<AccruingWeight> knownWeights = new HashSet<AccruingWeight>();
	
	private ConvolutionLayer(int rows, int cols) {
		nodeMap = new ConvolutionNode[rows][cols];
	}
	
	public static ConvolutionLayer create(int rows, int cols, ActivationFunction actFn) {
		ConvolutionLayer result = new ConvolutionLayer(rows, cols);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				result.addNode(new ConvolutionNode(actFn, result, r, c));
			}
		}
		result.dimensions = new Dimension(cols, rows);
		return result;
	}
	
	public static ConvolutionLayer createNewInputLayer(int rows, int cols) {
		return create(rows, cols, ActivationFunction.LINEAR);
	}

	public static ConvolutionLayer createNewSubsampleLayer(ConvolutionLayer inputLayer) {
		int subsampleRsz = 2;
		int subsampleCsz = 2;
		return createHiddenFromInputLayer(inputLayer, subsampleRsz, subsampleCsz, 0, 0);
	}
	
//	public void connectToNewInputLayer(ConvolutionLayer inputLayer, double overlap) {
//		final int imgH = inputLayer.getNumRows();
//		final int imgW = inputLayer.getNumCols();
//		final int nr = this.getNumRows();
//		final int nc = this.getNumCols();
//		final int patchHeight = (int) Math.round((1.0 + overlap * (nr - 1)) * imgH / nr);
//		final int patchWidth = (int) Math.round((1.0 + overlap * (nc - 1)) * imgW / nc);
//	}
	
	public static ConvolutionLayer createHiddenFromInputLayer(ConvolutionLayer inputLayer,
			int patchRSize, int patchCSize, int verticalOverlap, int horizontalOverlap) {
		ConvolutionLayer result = new ConvolutionLayer(1000, 1000);
		AccruingWeight[][] patchWeights = new AccruingWeight[patchRSize][patchCSize];
		for (int r = 0; r < patchRSize; r++) {
			for (int c = 0; c < patchCSize; c++) {
				AccruingWeight w = new AccruingWeight();
				patchWeights[r][c] = w;
				result.knownWeights.add(w);
			}
		}
		final Dimension inputDimension = inputLayer.getDimensions();
		int inR = 0, r = 0, c = 0;
		while (inR < inputDimension.getHeight()) {
			int inC = 0; c = 0;
			while (inC < inputDimension.getWidth()) {
				ConvolutionNode node = new ConvolutionNode(DefaultParameters.DEFAULT_ACT_FN, result, r, c);
				result.addNode(node);
				for (int rr = 0; rr < patchRSize; rr++) {
					if (inR + rr >= inputDimension.getHeight()) break;
					for (int cc = 0; cc < patchCSize; cc++) {
						if (inC + cc >= inputDimension.getWidth()) break;
						AccruingWeight weight = patchWeights[rr][cc];
						Connection.quickCreate(inputLayer.getNode(inR + rr, inC + cc), node, weight);
					}
				}
				inC += patchCSize - horizontalOverlap;
				c++;
			}
			inR += patchRSize - verticalOverlap;
			r++;
		}
		result.cropNodes(r, c);

		result.biasNode.connectToLayer(result);
		result.dimensions = new Dimension(c, r);
		return result;
	}
	
	private void cropNodes(int rows, int cols) {
		final ConvolutionNode[][] tmpMap = new ConvolutionNode[rows][cols];
		for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
			tmpMap[r][c] = nodeMap[r][c];
		}
		nodeMap = tmpMap;
	}

	protected void addNode(ConvolutionNode n) {
		getNodes().add(n);
		nodeMap[n.getR()][n.getC()] = n;
	}
	
	public ConvolutionNode getNode(int r, int c) {
		return nodeMap[r][c];
	}
	
	public Dimension getDimensions() {
		return this.dimensions;
	}
	
	public int getNumRows() {
		return this.dimensions.height;
	}
	
	public int getNumCols() {
		return this.dimensions.width;
	}
	
	public void clamp(BufferedImage img) {
		for (int r = 0; r < nodeMap.length; r++) {
			ConvolutionNode[] row = nodeMap[r];
			for (int c = 0; c < row.length; c++) {
				if (r >= img.getHeight() || c >= img.getWidth()) row[c].clamp(0);
				else {
					int rgb = img.getRGB(c, r);
					row[c].clamp(luminance(rgb) / 255.0);
				}
			}
		}
	}
	
	public static double luminance(int rgb) {
		int R = (rgb >> 16) & 0xFF;
		int G = (rgb >> 8) & 0xFF;
		int B = (rgb & 0xFF);
		return (R+R+R+B+G+G+G+G)>>3;
	}
	
	public BufferedImage getImage() {
		BufferedImage result = new BufferedImage(getNumCols(), getNumRows(), BufferedImage.TYPE_INT_RGB);
		
		for (int r = 0; r < getNumRows(); r++) {
			for (int c = 0; c < getNumCols(); c++){
				int grayLevel = (int) Math.round(255.0 * nodeMap[r][c].getActivation());
				int gray = (grayLevel << 16) + (grayLevel << 8) + grayLevel; 
				result.setRGB(c, r, gray);
	    	}
	    }
	    return result;
	}
}
