package ann.convolution;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.*;

import ann.*;


public class ConvolutionNetwork implements NeuralNetwork {

	private final ConvolutionLayer inputLayer;
	private final Layer<? extends Node> outputLayer;

	private final ArrayList<ArrayList<ConvolutionLayer>> hiddenLayers = new ArrayList<ArrayList<ConvolutionLayer>>();

	public ConvolutionNetwork(int inR, int inC, ConvSpec... convspecs) {
//		BiasNode.clearConnections();
		inputLayer = ConvolutionLayer.createNewInputLayer(inR, inC);
		Collection<ConvolutionLayer> lastLayers = new ArrayList<ConvolutionLayer>();
		lastLayers.add(getInputLayer());
		for (ConvSpec cs : convspecs) {
			Collection<ConvolutionLayer> nextLayers = new ArrayList<ConvolutionLayer>();
			ArrayList<ConvolutionLayer> hl = new ArrayList<ConvolutionLayer>();
			ArrayList<ConvolutionLayer> hlss = new ArrayList<ConvolutionLayer>();
			hiddenLayers.add(hl);
			hiddenLayers.add(hlss);
			for (ConvolutionLayer lastLayer : lastLayers) {
				for (int i = 0; i < cs.getNumHorizontalLayers(); i++) {
					System.out.println("making convo layer");
					final ConvolutionLayer cl = ConvolutionLayer.createHiddenFromInputLayer(lastLayer,
							cs.getPatchH(), cs.getPatchW(), cs.getOverlapH(), cs.getOverlapW());
					System.out.println("making subsample layer");
					final ConvolutionLayer ss = ConvolutionLayer.createNewSubsampleLayer(cl);
					hl.add(cl);
					hlss.add(ss);
					nextLayers.add(ss);
				}
			}
			lastLayers = nextLayers;
		}
		Collection<ConvolutionNode> lastNodes = new HashSet<ConvolutionNode>();
		for (ConvolutionLayer cl : lastLayers) lastNodes.addAll(cl.getNodes());

		Layer<? extends Node> nextToLastLayer = Layer.createHiddenFromInputLayer(lastNodes, 10,
				ActivationFunction.SIGMOID0p5, Node.BASIC_NODE_FACTORY);
		outputLayer = Layer.createHiddenFromInputLayer(nextToLastLayer.getNodes(), 5,
				ActivationFunction.SIGMOID0p5, Node.BASIC_NODE_FACTORY);
	}

	public ConvolutionLayer getInputLayer() {
		return inputLayer;
	}

	public Layer<? extends Node> getOutputLayer() {
		return outputLayer;
	}

	public Collection<ConvolutionLayer> getHiddenLayers(int h) {
		return hiddenLayers.get(h);
	}
	
	public ConvolutionLayer getHiddenLayer(int h, int i) {
		return hiddenLayers.get(h).get(i);
	}

	public BufferedImage getImage() {
		BufferedImage result = new BufferedImage(5000, 5000, BufferedImage.TYPE_INT_RGB);
		int x = 0;
		for (int h = 0; h < hiddenLayers.size(); h += 2) {
			ArrayList<ConvolutionLayer> layers = hiddenLayers.get(h);
			ArrayList<ConvolutionLayer> ssLayers = hiddenLayers.get(h + 1);
			int y = 0, w = 0;
			for (int i = 0; i < layers.size(); i++) {
				BufferedImage img = layers.get(i).getImage();
				BufferedImage imgss = ssLayers.get(i).getImage();
				Graphics2D g = result.createGraphics();
				g.drawImage(img, x, y, null);
				g.drawImage(imgss, x + img.getWidth(), y + (img.getHeight() - imgss.getHeight()) / 2, null);
				y += img.getHeight();
				w = Math.max(w, img.getWidth() + imgss.getWidth());
			}
			x += w;
		}
	    return result;
	}

	@Override
	public ArrayList<? extends Node> getInputNodes() {
		return getInputLayer().getNodes();
	}

	@Override
	public ArrayList<? extends Node> getOutputNodes() {
		return getOutputLayer().getNodes();
	}

}
