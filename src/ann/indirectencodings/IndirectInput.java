package ann.indirectencodings;

public class IndirectInput {

	private final String name;
	private final double[] vector;

	public IndirectInput(String name, double... vector) {
		this.name = name;
		this.vector = vector;
	}
	
	@Override
	public String toString() {
		return name;
	}

	public double[] getVector() {
		return vector;
	}
}
