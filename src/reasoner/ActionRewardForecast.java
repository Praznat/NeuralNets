package reasoner;

public class ActionRewardForecast {
	private final double action[];
	private final double reward;
	private final Forecast forecast;
	public ActionRewardForecast(double[] action, double reward, Forecast forecast) {
		this.action = action;
		this.reward = reward;
		this.forecast = forecast;
	}
	public double[] getAction() {
		return action;
	}
	public double getReward() {
		return reward;
	}
	public Forecast getForecast() {
		return forecast;
	}
	@Override
	public String toString() {
		return action + "	" + forecast;
	}
}
