//! Offline batch bridge: actual Struktura `dfa_short` on supplied numeric windows.
//! Input CSV: x,y,v0,...,vN (N>=32). Empty or nonfinite => abstain.
//! Output CSV: x,y,alpha,r_squared,confidence. No label/source coordinates accepted.
//! This is NOT Guard, nor a fault/health classifier.
use std::io::{self, BufRead};
use struktura::dfa_short;
fn main() {
    println!("x,y,alpha,r_squared,confidence");
    for (line_no, line) in io::stdin().lock().lines().enumerate() {
        let line = match line { Ok(v) => v, Err(e) => {eprintln!("read error: {e}"); std::process::exit(2)} };
        if line.trim().is_empty() {continue;}
        let f:Vec<&str>=line.split(',').collect();
        if f.len()<34 {eprintln!("line {} expected x,y and >=32 readings",line_no+1);std::process::exit(2)}
        let x=match f[0].parse::<usize>() {Ok(v)=>v,Err(_)=>{eprintln!("bad x");std::process::exit(2)}};
        let y=match f[1].parse::<usize>() {Ok(v)=>v,Err(_)=>{eprintln!("bad y");std::process::exit(2)}};
        let values:Option<Vec<f64>>=f[2..].iter().map(|s| s.parse::<f64>().ok().filter(|v|v.is_finite())).collect();
        let Some(values)=values else {println!("{x},{y},,,0");continue};
        match dfa_short(&values) {
           Some(d) if d.alpha.is_finite()&&d.r_squared.is_finite()&&d.r_squared>=0.5&&d.r_squared<=1.00001 => {
             println!("{x},{y},{:.10},{:.10},1",d.alpha,d.r_squared);
           }
           _=> println!("{x},{y},,,0"),
        }
    }
}
