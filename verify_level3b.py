"""Run the validated 3 nm, longitudinal Level-3B verification case."""

from fullwave_calibration.calibration import solve_case


def main() -> None:
    field, history = solve_case(3.0, 0.0)
    if field is None:
        raise RuntimeError("the verification case failed multipole convergence")
    ex = field[0]
    print(f"Ex.real: {ex.real:.12g}")
    print(f"Ex.imag: {ex.imag:.12g}")
    print(f"abs(Ex): {abs(ex):.12g}")
    print(f"phase(Ex): {__import__('numpy').angle(ex):.12g}")
    print(f"abs(Ex/E0): {abs(ex):.12g}")
    print(f"lmax: {history[-1]['lmax']}")


if __name__ == "__main__":
    main()
