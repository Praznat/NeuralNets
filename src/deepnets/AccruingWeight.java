package deepnets;

public class AccruingWeight implements Weight {
	
	private double weight, newWeight, lastWgtChg, blameFromOutput;

	public AccruingWeight() {
		this(DefaultParameters.RANDOM_WEIGHT());
	}
	public AccruingWeight(double w) {
		this.weight = w;
		this.newWeight = w;
		this.lastWgtChg = 0;
	}
	public void enactWeightChange() {
		this.weight = this.newWeight;
	}
	@Override
	public double getWeight() {
		return weight;
	}
	public void setWeight(double weight) {
		this.newWeight = weight;
	}
	public void chgWeight(double inc) {
		this.newWeight += inc;
	}
	public void multWeight(double mult) {
		this.newWeight *= mult;
	}
	public double getBlameFromOutput() {
		return blameFromOutput;
	}
	public void propagateError(double delta, double inputActivation, double learningRate, double momentum, boolean isFrozen) {
		this.blameFromOutput = delta * weight;
		double change = learningRate * delta * inputActivation + momentum * this.lastWgtChg;
		if (!isFrozen) chgWeight(change);
		this.lastWgtChg = change;
	}

}
