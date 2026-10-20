//! Deterministic paired-seed benchmark. Outputs per-episode CSV to stdout.
use colony_exploration::{run_episode, Config, Policy, World};
use std::env;

fn read_arg(args: &[String], name: &str, default: usize) -> Result<usize, String> {
    match args.iter().position(|arg| arg == name) {
        Some(index) => args.get(index + 1)
            .ok_or_else(|| format!("missing value for {name}"))?
            .parse::<usize>()
            .map_err(|_| format!("invalid integer for {name}")),
        None => Ok(default),
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("colony-bench [--seeds N] [--width N] [--height N] [--radius N] [--max-steps N]");
        println!("Prints paired policy episodes as CSV. No hardware control or SLAM.");
        return Ok(());
    }
    let seeds = read_arg(&args, "--seeds", 12)?;
    let width = read_arg(&args, "--width", 25)?;
    let height = read_arg(&args, "--height", 19)?;
    let config = Config {
        radius: read_arg(&args, "--radius", 3)?,
        max_steps: read_arg(&args, "--max-steps", 3000)?,
    };
    if seeds == 0 || width < 3 || height < 3 {
        return Err("invalid seeds or dimensions".into());
    }
    println!("seed,policy,coverage,steps,decisions,growth_nodes,collisions,completed,stopped_for_safety");
    let mut totals = [0usize; 2];
    let mut completed = [0usize; 2];
    for seed in 0..seeds {
        let world = World::generated(seed as u64, width, height);
        for (index, policy) in [Policy::Frontier, Policy::Growth].iter().copied().enumerate() {
            let result = run_episode(&world, policy, config)?;
            println!("{},{},{:.6},{},{},{},{},{},{}",
                seed, policy.name(), result.coverage, result.steps, result.decisions,
                result.growth_nodes, result.collisions, result.completed, result.stopped_for_safety);
            totals[index] += result.steps;
            completed[index] += usize::from(result.completed);
        }
    }
    eprintln!("frontier: {:.2} mean steps, {}/{} complete",
        totals[0] as f64 / seeds as f64, completed[0], seeds);
    eprintln!("growth:   {:.2} mean steps, {}/{} complete",
        totals[1] as f64 / seeds as f64, completed[1], seeds);
    Ok(())
}
