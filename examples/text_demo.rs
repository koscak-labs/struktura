use struktura::text::{text_structure, sentence_lengths};

fn main() {
    // Test 1: Human literary prose (Melville style - varied sentence length)
    let human_literary = "Call me Ishmael. Some years ago — never mind how long precisely — having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people's hats off — then, I account it high time to get to sea as soon as I can. This is my substitute for pistol and ball. With a philosophical flourish Cato throws himself upon his sword; I quietly take to the ship. There is nothing surprising in this. If they but knew it, almost all men in their degree, some time or other, cherish very nearly the same feelings towards the ocean with me.";

    // Test 2: Technical/uniform (deliberately even sentence lengths)
    let technical = "The system processes input data. It validates each record. The pipeline runs sequentially. Each stage transforms the data. Results are stored in memory. The cache improves performance. Errors are logged to disk. The monitor checks health. Status is reported upstream. The dashboard shows metrics. Alerts trigger on thresholds. The operator reviews alerts. Actions are taken promptly. The system recovers gracefully. Logs are rotated daily. Backups run at midnight. The database is replicated. Failover is automatic. Recovery takes seconds. The architecture is proven. Testing covers edge cases. Documentation is maintained. The team reviews changes. Deployment is automated. Monitoring never stops. The cycle continues endlessly. Performance meets targets. Reliability exceeds goals. The system serves users. Everyone is satisfied. Quality is paramount. Security is enforced. Compliance is verified. Audits pass regularly. The infrastructure scales. Growth is anticipated. Planning never stops. The future looks promising. Innovation drives progress. Success breeds confidence. Confidence enables ambition. Ambition fuels growth. Growth demands infrastructure. Infrastructure requires planning. Planning needs data. Data comes from monitoring. Monitoring watches everything. Everything connects. Nothing is isolated. The system lives. Life continues. Growth persists. Change is constant. Adaptation is key. Survival requires evolution. Evolution takes time. Time moves forward. Forward is the way. The way is clear. Clarity enables action. Action produces results. Results prove value. Value justifies investment. Investment enables capability. Capability drives success.";

    // Test 3: Random-ish (very short, choppy)
    let choppy = "Go. Stop. Wait. Run. Jump. Fall. Rise. Walk. Sit. Stand. Look. See. Hear. Feel. Touch. Taste. Smell. Think. Know. Learn. Grow. Change. Move. Stay. Leave. Come. Try. Fail. Win. Lose. Start. End. Live. Die. Love. Hate. Give. Take. Make. Break. Build. Burn. Sing. Dance. Play. Work. Rest. Sleep. Wake. Eat. Drink. Read. Write. Speak. Listen. Watch. Wait. Hope. Dream. Plan. Act. Do. Be.";

    println!("STRUKTURA TEXT STRUCTURE ANALYSIS");
    println!("================================\n");

    for (name, text) in [
        ("Human literary prose", human_literary),
        ("Technical uniform", technical),
        ("Choppy minimal", choppy),
    ] {
        let result = text_structure(text);
        let lens = &result.sentence_lengths;
        println!("  {}", name);
        println!("    sentences: {}  mean_len: {:.1} chars", result.sentence_count, result.mean_sentence_len);
        println!("    DFA alpha: {:.3}  R²: {:.4}", result.dfa.alpha, result.dfa.r_squared);
        println!("    quality: {}", result.law.quality);
        if lens.len() <= 20 {
            println!("    lengths: {:?}", lens);
        } else {
            println!("    lengths: [{}, {}, {}, ... {} more ... {}, {}, {}]",
                lens[0], lens[1], lens[2], lens.len() - 6,
                lens[lens.len()-3], lens[lens.len()-2], lens[lens.len()-1]);
        }
        println!();
    }
}
