package reasoner;

import java.io.Serializable;
import java.util.Arrays;

@SuppressWarnings("serial")
public class DiscreteState implements Serializable {
	final double[] rawState;
	final boolean[] factoredState;
	
	public DiscreteState(double[] rawState) {
		this.rawState = new double[rawState.length];
		this.factoredState = new boolean[rawState.length];
		for (int i = 0; i < rawState.length; i++) {
			this.factoredState[i] = rawState[i] > 0.5;
			this.rawState[i] = Math.round(rawState[i]);
		}
	}
	@Override
	public boolean equals(Object other) {
		DiscreteState o = (DiscreteState) other;
		boolean result = true;
		if (this.factoredState.length != o.factoredState.length) result = false;
		else for (int i = 0; i < factoredState.length; i++)
			if (this.factoredState[i] != o.factoredState[i]) {
				result = false;
				break;
			}
//		if (other.toString().equals(this.toString()) && !result) {
//			System.out.println("THIS IS CUZ OF LENGTH DIFFERENCE");
//		}
		return result;
	}
	@Override
	public int hashCode() {
		return Arrays.hashCode(factoredState);
	}
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i< factoredState.length; i++) if (factoredState[i]) sb.append(i+".");
		return sb.toString();
	}
	public double[] getRawState() {
		return rawState;
	}
}
