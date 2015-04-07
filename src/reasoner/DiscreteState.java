package reasoner;

import java.util.Arrays;

public class DiscreteState {
	final double[] rawState;
	final boolean[] factoredState;
	
	public DiscreteState(double[] rawState) {
		this.rawState = rawState;
		this.factoredState = new boolean[rawState.length];
		for (int i = 0; i < rawState.length; i++) this.factoredState[i] = rawState[i] > 0.5;
	}
	@Override
	public boolean equals(Object other) {
		DiscreteState o = (DiscreteState) other;
		if (this.factoredState.length != o.factoredState.length) return false;
		for (int i = 0; i < factoredState.length; i++)
			if (this.factoredState[i] != o.factoredState[i]) return false;
		return true;
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
