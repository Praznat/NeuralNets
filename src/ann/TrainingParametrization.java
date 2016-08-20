package ann;

import java.io.Serializable;

@SuppressWarnings("serial")
public class TrainingParametrization implements Serializable {

	private final int[] numHidden;
	private final int epochs;
	private final double lRate;
	private final double mRate;
	private final double sRate;

	public TrainingParametrization(int[] numHidden, int epochs, double lRate, double mRate, double sRate) {
		this.numHidden = numHidden;
		this.epochs = epochs;
		this.lRate = lRate;
		this.mRate = mRate;
		this.sRate = sRate;
	}

	public int[] getNumHidden() {
		return numHidden;
	}

	public int getEpochs() {
		return epochs;
	}

	public double getlRate() {
		return lRate;
	}

	public double getmRate() {
		return mRate;
	}

	public double getsRate() {
		return sRate;
	}
}
