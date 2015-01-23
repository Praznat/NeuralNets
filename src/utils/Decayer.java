package utils;

public class Decayer {
	public static final double LOG_HALF = Math.log(0.5);
	
	private final long halfLifeMs;
	private long lastMs;
	public Decayer(long decayFactor) {
		this.halfLifeMs = decayFactor;
	}
	/** *
	 * returns post-decay value of weightedScore, decaying more as ms goes further from the last ms decay happened
	 * @param weightedScore
	 * @param ms
	 * @return
	 */
	public double decayTo(double weightedScore, long ms) {
		if (weightedScore != 0) {
			long elapsed = ms - lastMs;
			weightedScore *= decayFactor(halfLifeMs, elapsed);
		}
		lastMs = ms;
		return weightedScore;
	}
	
	/**
	 * returns weighted average of newObservation and oldEMA, with higher weight
	 * going to newObservation as time (ms) since last decay increases
	 * 
	 * best used in form: ema = this.newEMA(newObs, ema, nowMs)
	 * @param newObservation
	 * @param oldEMA
	 * @param ms
	 * @return
	 */
	public double newEMA(double newObservation, double oldEMA, long ms) {
		long elapsed = ms - lastMs;
		final double decayWgt = decayFactor(halfLifeMs, elapsed);
		lastMs = ms;
		return newObservation * (1 - decayWgt) + oldEMA * decayWgt;
	}
	
	private static double decayFactor(double halflife, double elapsed) {
		return Math.exp(elapsed * LOG_HALF / halflife);
	}
}
