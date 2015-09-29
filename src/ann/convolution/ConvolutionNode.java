package ann.convolution;

import ann.*;

public class ConvolutionNode extends Node {

	private final int r, c;
	
	public ConvolutionNode(ActivationFunction activationFunction, ConvolutionLayer parentLayer, int r, int c) {
		super(activationFunction, parentLayer, r + "," + c);
		this.r = r;
		this.c = c;
	}

	public int getR() {
		return r;
	}

	public int getC() {
		return c;
	}

}
