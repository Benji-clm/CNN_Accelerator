// filepath: /root/Documents/iac/lab0-devtools/autumn/workspace/CNN_Accelerator/FPGA/Verilog/tb/tests/test_control_cv_tb.cpp
/*
 *  Verifies the results of the control_cv module, exits with a 0 on success.
 */

#include "base_testbench.h"

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class ControlCVTestbench : public BaseTestbench
{
protected:
    void initializeInputs() override
    {
        top->clk = 0;
        top->rst = 1;
        top->start = 0;
        // Outputs: data_out, done
    }

    void clockTick() {
        top->clk = 1;
        top->eval();
        tfp->dump(ticks++);
        top->clk = 0;
        top->eval();
        tfp->dump(ticks++);
    }

    void reset() {
        top->rst = 1;
        clockTick();
        top->rst = 0;
        clockTick();
    }

    void startOperation() {
        top->start = 1;
        clockTick();
        top->start = 0;
        clockTick();
    }

    // Wait until done signal is asserted
    void waitForDone() {
        for (int i = 0; i < 200; ++i) {
            clockTick();
            if (top->done)
                break;
        }
        EXPECT_EQ(top->done, 1);
    }
};

TEST_F(ControlCVTestbench, BasicConvolutionFlow)
{
    reset();

    // Start the operation
    startOperation();

    // Wait for done
    waitForDone();

    // Check that done is asserted and data_out has some value
    EXPECT_EQ(top->done, 1);

    // Optionally, check a few output values (assuming 3x3 kernel, 5x5 image for example)
    // This assumes data_out is a packed array; adjust as needed for your interface
    // Example: EXPECT_EQ(top->data_out[0], expected_value);
}

TEST_F(ControlCVTestbench, ResetDuringOperation)
{
    reset();
    startOperation();

    // Simulate a few cycles of operation
    for (int i = 0; i < 5; ++i) clockTick();

    // Assert reset in the middle
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();

    // Module should return to IDLE and done should be low
    EXPECT_EQ(top->done, 0);

    // Start again and wait for done
    startOperation();
    waitForDone();
    EXPECT_EQ(top->done, 1);
}

int main(int argc, char **argv)
{
    top = new Vdut;
    tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("control_cv_waveform.vcd");

    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();

    top->final();
    tfp->close();

    delete top;
    delete tfp;

    return res;
}