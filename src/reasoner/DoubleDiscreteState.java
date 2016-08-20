package reasoner;

import java.io.Serializable;
import java.util.Arrays;

import ann.Utils;

@SuppressWarnings("serial")
public class DoubleDiscreteState implements Serializable {
	final DiscreteState ds1, ds2;
	
	public DoubleDiscreteState(double[] inputs, double[] outputs) {
		this.ds1 = new DiscreteState(inputs);
		this.ds2 = new DiscreteState(outputs);
	}
	@Override
	public boolean equals(Object other) {
		DoubleDiscreteState o = (DoubleDiscreteState) other;
		return this.ds1.equals(o.ds1) && this.ds2.equals(o.ds2);
	}
	@Override
	public int hashCode() {
		return Arrays.hashCode(Utils.concat(ds1.rawState, ds2.rawState));
	}
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(ds1).append("	->	").append(ds2);
		return sb.toString();
	}
}
