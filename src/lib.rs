pub mod card;
pub mod pdcfr;
pub mod tree;

#[cfg(feature = "python")]
mod python_bindings {
    use crate::card::Card;
    use crate::tree::SubgameSolver;
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
    use pyo3::prelude::*;

    #[pyclass]
    pub struct PySubgameSolver {
        inner: SubgameSolver,
    }

    #[pymethods]
    impl PySubgameSolver {
        #[new]
        pub fn new(
            board_strs: Vec<String>,
            oop_range: &str,
            ip_range: &str,
            starting_pot: f32,
            effective_stack: f32,
            bet_sizes: Vec<f32>,
        ) -> PyResult<Self> {
            let mut board = Vec::with_capacity(board_strs.len());
            for s in board_strs {
                let c = Card::from_str(&s).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid card: {}", s))
                })?;
                board.push(c);
            }
            let inner = SubgameSolver::new(
                &board,
                oop_range,
                ip_range,
                starting_pot,
                effective_stack,
                &bet_sizes,
            );
            Ok(Self { inner })
        }

        #[staticmethod]
        pub fn from_arrays(
            board_ids: Vec<u8>,
            oop_range: PyReadonlyArray1<f32>,
            ip_range: PyReadonlyArray1<f32>,
            starting_pot: f32,
            effective_stack: f32,
            bet_sizes: Vec<f32>,
        ) -> PyResult<Self> {
            let board: Vec<Card> = board_ids.into_iter().map(Card).collect();
            let oop_slice = oop_range.as_slice()?;
            let ip_slice = ip_range.as_slice()?;

            let inner = SubgameSolver::from_ranges(
                &board,
                oop_slice,
                ip_slice,
                starting_pot,
                effective_stack,
                &bet_sizes,
            );
            Ok(Self { inner })
        }

        pub fn solve<'py>(
            &mut self,
            py: Python<'py>,
            iterations: u32,
        ) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>) {
            let (oop_cfvs, ip_cfvs) = py.allow_threads(|| self.inner.solve(iterations));
            (
                oop_cfvs.into_pyarray_bound(py),
                ip_cfvs.into_pyarray_bound(py),
            )
        }

        pub fn get_root_strategy<'py>(
            &self,
            py: Python<'py>,
            hand_str: &str,
        ) -> PyResult<Bound<'py, PyArray1<f32>>> {
            if hand_str.len() != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Hand must be 4 chars like 'AcAh'",
                ));
            }
            let c1 = Card::from_str(&hand_str[0..2]).unwrap();
            let c2 = Card::from_str(&hand_str[2..4]).unwrap();
            let h_mask = c1.mask() | c2.mask();

            let idx = self
                .inner
                .valid_hands
                .iter()
                .position(|h| h.mask == h_mask)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Hand blocked or not in range")
                })?;

            let strat = self.inner.nodes.first().unwrap().hand_nodes[idx].get_average_strategy();
            Ok(strat.into_pyarray_bound(py))
        }
    }

    #[pymodule]
    fn apex_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PySubgameSolver>()?;
        Ok(())
    }
}
