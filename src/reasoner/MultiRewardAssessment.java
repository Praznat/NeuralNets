package reasoner;

public class MultiRewardAssessment implements StateObserver {

	private final RewardFunction rewardFn;
	private final double discountRate;
	private double sumReward;
	private double sumRealism;
	
	public MultiRewardAssessment(RewardFunction rewardFn, double discountRate) {
		this.rewardFn = rewardFn;
		this.discountRate = discountRate;
	}
	
	@Override
	public void observeState(int t, double realism, double... state) {
		sumReward += discount(rewardFn.getReward(state), t) * realism;
		sumRealism += realism;
	}
	
	public double getExpReward() {
		return sumReward / sumRealism;
	}
	
	private double discount(double d, int t) {
		return d * Math.exp(-discountRate * t);
	}
}
