#!/bin/bash
# Check what's actually in your server's train_loso_functional_qat.py file

echo "Checking dense_filters definition in train_loso_functional_qat.py..."
echo "======================================================================="
echo ""

# Show the actual lines around dense_filters definition
sed -n '399,410p' train_loso_functional_qat.py

echo ""
echo "======================================================================="
echo "Expected (SAFE):"
echo '    dense_filters = args.quantize_dense_names or ['
echo '        "fc1", "fc2",          # FFN layers only'
echo '        # "proj_x1", "proj_y1",  # Projection (optional, can add if needed)'
echo '        # "q_proj", "k_proj", "v_proj", "out_proj",  # ATTENTION PROJECTIONS - DO NOT QUANTIZE!'
echo '    ]'
echo ""
echo "If your file shows UNCOMMENTED attention projections, that's the problem!"
echo "======================================================================="

