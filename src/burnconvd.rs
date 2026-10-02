use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d,
        Relu,
        conv::{Conv1d, Conv1dConfig},
        loss::CrossEntropyLossConfig,
        pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{
        ElementConversion, Int, Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

/*
Gaurav Sablok
gsablok@proton.me
 */

const AMINO_ACIDS: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
const PAD_TOKEN: usize = 20;
const VOCAB_SIZE: usize = 21; // 20 amino acids + pad
const NUM_CLASSES: usize = 2; // e.g. "not hydrophobic-rich" vs "hydrophobic-rich"

pub fn aa_to_id(aa: u8) -> usize {
    AMINO_ACIDS
        .iter()
        .position(|&c| c == aa)
        .unwrap_or(PAD_TOKEN)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ProteinConv1d<B: Backend> {
    embedding: Embedding<B>,
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    pool: AdaptiveAvgPool1d,
    dropout: Dropout,
    activation: Relu,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

#[derive(Debug, Clone)]
pub struct ProteinConv1dConfig {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub conv_channels: usize,
    pub kernel_size: usize,
    pub hidden_size: usize,
    pub num_classes: usize,
    pub dropout: f64,
}

impl Default for ProteinConv1dConfig {
    fn default() -> Self {
        Self {
            vocab_size: VOCAB_SIZE,
            embedding_dim: 32,
            conv_channels: 64,
            kernel_size: 5,
            hidden_size: 64,
            num_classes: NUM_CLASSES,
            dropout: 0.2,
        }
    }
}

impl ProteinConv1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ProteinConv1d<B> {
        // "Same" padding keeps the sequence length constant across conv layers,
        // which is convenient since protein sequences vary in length and we
        // don't want to hand-track output sizes.
        let conv1 = Conv1dConfig::new(self.embedding_dim, self.conv_channels, self.kernel_size)
            .with_padding(PaddingConfig1d::Same)
            .init(device);
        let conv2 = Conv1dConfig::new(self.conv_channels, self.conv_channels, self.kernel_size)
            .with_padding(PaddingConfig1d::Same)
            .init(device);

        ProteinConv1d {
            embedding: EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device),
            conv1,
            conv2,
            pool: AdaptiveAvgPool1dConfig::new(1).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Relu::new(),
            fc1: LinearConfig::new(self.conv_channels, self.hidden_size).init(device),
            fc2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> ProteinConv1d<B> {
    /// `tokens`: [batch, seq_len] amino-acid token ids.
    /// Returns logits of shape [batch, num_classes].
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        // [batch, seq_len] -> [batch, seq_len, embedding_dim]
        let x = self.embedding.forward(tokens);

        // Conv1d expects [batch, channels, length], so swap the last two dims.
        let x = x.swap_dims(1, 2); // -> [batch, embedding_dim, seq_len]

        let x = self.activation.forward(self.conv1.forward(x));
        let x = self.dropout.forward(x);
        let x = self.activation.forward(self.conv2.forward(x));

        // Global average pooling over the sequence dimension:
        // [batch, conv_channels, seq_len] -> [batch, conv_channels, 1]
        let x = self.pool.forward(x);
        // burn 0.21: squeeze() takes no runtime dim argument anymore; the
        // dimension to drop is inferred from the target rank via the
        // const generic (here going from 3D -> 2D).
        let x = x.squeeze::<2>(); // -> [batch, conv_channels]

        let x = self.activation.forward(self.fc1.forward(x));
        let x = self.dropout.forward(x);
        self.fc2.forward(x) // -> [batch, num_classes]
    }

    /// Convenience wrapper that also computes cross-entropy loss against labels.
    pub fn forward_loss(
        &self,
        tokens: Tensor<B, 2, Int>,
        labels: Tensor<B, 1, Int>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let logits = self.forward(tokens);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), labels);
        (logits, loss)
    }
}

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------

/// A single (sequence, label) example held in plain Rust types before it is
/// turned into tensors. Keeping this intermediate representation makes it
/// easy to shuffle/split/batch on the CPU without touching the backend.
#[derive(Debug, Clone)]
pub struct ProteinExample {
    pub sequence: Vec<u8>,
    pub label: i64,
}

/// Loads a dataset from a plain-text file at `path`.
///
/// Expected format: one example per line, comma-separated:
///
/// ```text
/// MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ,1
/// ACDEFGHIKLMNPQRSTVWY...,0
/// ```
///
/// Blank lines and lines starting with `#` are skipped. Any amino acid
/// character not in the 20-letter alphabet is mapped to the pad token by
/// `aa_to_id`.
pub fn load_dataset<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<ProteinExample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut examples = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.rsplitn(2, ',');
        let label_str = parts.next().unwrap_or("").trim();
        let seq_str = parts.next().unwrap_or("").trim();

        if seq_str.is_empty() {
            continue;
        }

        let label: i64 = match label_str.parse() {
            Ok(v) => v,
            Err(_) => continue, // skip malformed rows instead of aborting the whole load
        };

        examples.push(ProteinExample {
            sequence: seq_str.as_bytes().to_vec(),
            label,
        });
    }

    Ok(examples)
}

/// Converts a slice of examples into `(tokens, labels)` tensors.
///
/// Sequences longer than `seq_len` are truncated; shorter ones are
/// right-padded with `PAD_TOKEN`.
pub fn examples_to_tensors<B: Backend>(
    examples: &[ProteinExample],
    seq_len: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 1, Int>) {
    let batch_size = examples.len();
    let mut token_data = Vec::with_capacity(batch_size * seq_len);
    let mut label_data = Vec::with_capacity(batch_size);

    for example in examples {
        for i in 0..seq_len {
            let id = match example.sequence.get(i) {
                Some(&aa) => aa_to_id(aa),
                None => PAD_TOKEN,
            };
            token_data.push(id as i64);
        }
        label_data.push(example.label);
    }

    let tokens = Tensor::<B, 1, Int>::from_data(token_data.as_slice(), device)
        .reshape([batch_size, seq_len]);
    let labels = Tensor::<B, 1, Int>::from_data(label_data.as_slice(), device);

    (tokens, labels)
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

pub fn accuracy<B: Backend>(logits: Tensor<B, 2>, labels: Tensor<B, 1, Int>) -> f32 {
    // burn 0.21: squeeze() infers the dropped dim from the target rank
    // (here going from 2D [batch, 1] -> 1D [batch]).
    let predictions = logits.argmax(1).squeeze::<1>();
    let correct = predictions
        .clone()
        .equal(labels)
        .int()
        .sum()
        .into_scalar()
        .elem::<f32>();
    correct / predictions.dims()[0] as f32
}

/// Trains the model on data read from `data_path`.
///
/// * `data_path` - path to a comma-separated `sequence,label` text file
///   (see [`load_dataset`] for the expected format).
/// * `seq_len` - fixed sequence length that every example is truncated or
///   padded to before being embedded.
pub fn train<B: AutodiffBackend>(device: B::Device, data_path: &str, seq_len: usize) {
    let config = ProteinConv1dConfig::default();
    let mut model: ProteinConv1d<B> = config.init(&device);
    let mut optimizer = AdamConfig::new().init();

    let mut rng = StdRng::seed_from_u64(42);
    let batch_size = 32;
    let learning_rate = 1e-3;
    let epochs = 20;

    let mut examples = load_dataset(data_path)
        .unwrap_or_else(|e| panic!("failed to load dataset from {data_path}: {e}"));
    assert!(
        !examples.is_empty(),
        "no usable examples found in {data_path}"
    );

    // Shuffle once up front, then hold out 20% for validation.
    examples.shuffle(&mut rng);
    let split_at = ((examples.len() as f32) * 0.8).round() as usize;
    let split_at = split_at.clamp(1, examples.len().saturating_sub(1).max(1));
    let (train_examples, val_examples) = examples.split_at(split_at);

    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0f32;
        let mut epoch_acc = 0.0f32;
        let mut steps = 0usize;

        // Shuffle the training indices fresh each epoch.
        let mut order: Vec<usize> = (0..train_examples.len()).collect();
        order.shuffle(&mut rng);

        for chunk in order.chunks(batch_size) {
            let batch: Vec<ProteinExample> =
                chunk.iter().map(|&i| train_examples[i].clone()).collect();
            let (tokens, labels) = examples_to_tensors::<B>(&batch, seq_len, &device);

            let (logits, loss) = model.forward_loss(tokens, labels.clone());
            epoch_loss += loss.clone().into_scalar().elem::<f32>();
            epoch_acc += accuracy(logits, labels);
            steps += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, grads);
        }

        println!(
            "epoch {:>2}/{epochs}  loss: {:.4}  accuracy: {:.1}%",
            epoch,
            epoch_loss / steps.max(1) as f32,
            100.0 * epoch_acc / steps.max(1) as f32
        );
    }

    // Held-out validation check on the examples set aside earlier.
    if !val_examples.is_empty() {
        let (tokens, labels) = examples_to_tensors::<B>(val_examples, seq_len, &device);
        let (logits, loss) = model.forward_loss(tokens, labels.clone());
        println!(
            "\nvalidation  loss: {:.4}  accuracy: {:.1}%",
            loss.into_scalar().elem::<f32>(),
            100.0 * accuracy(logits, labels)
        );
    } else {
        println!("\nno validation examples available (dataset too small)");
    }
}
