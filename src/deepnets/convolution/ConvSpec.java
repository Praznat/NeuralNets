package deepnets.convolution;

public class ConvSpec {

	private final int numHorizontalLayers;
	private final int patchH, patchW;
	private final int overlapH, overlapW;
	
	public ConvSpec(int numHorizontalLayers, int patchH, int patchW, int overlapH, int overlapW) {
		this.numHorizontalLayers = numHorizontalLayers;
		this.patchH = patchH;
		this.patchW = patchW;
		this.overlapH = overlapH;
		this.overlapW = overlapW;
	}

	public int getNumHorizontalLayers() {
		return numHorizontalLayers;
	}

	public int getPatchH() {
		return patchH;
	}

	public int getPatchW() {
		return patchW;
	}

	public int getOverlapH() {
		return overlapH;
	}

	public int getOverlapW() {
		return overlapW;
	}

}
