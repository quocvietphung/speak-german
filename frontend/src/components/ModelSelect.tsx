"use client";

import * as React from "react";
import {
    Field,
    NativeSelect,
    Box,
    Text,
    HStack,
    Badge,
} from "@chakra-ui/react";

type ModelValue = "base" | "fine_tuned";

export interface ModelSelectProps {
    modelId: ModelValue;
    onChange: (value: ModelValue) => void;
}

export default function ModelSelect({modelId, onChange}: ModelSelectProps) {
    const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        onChange(e.currentTarget.value as ModelValue);
    };

    const label = modelId === "base" ? "whisper_tiny_de" : "whisper_tiny_de_finetuned";

    return (
        <Box>
            <Field.Root>
                <HStack justify="space-between" mb="1">
                    <Field.Label>Model</Field.Label>
                    <Badge colorPalette="teal" variant="subtle">{label}</Badge>
                </HStack>

                <NativeSelect.Root size="sm" width="280px">
                    <NativeSelect.Field
                        value={modelId}
                        onChange={handleChange}
                        aria-label="Select ASR model"
                    >
                        <option value="base">whisper_tiny_de</option>
                        <option value="fine_tuned">whisper_tiny_de_finetuned</option>
                    </NativeSelect.Field>
                    <NativeSelect.Indicator/>
                </NativeSelect.Root>

                <Field.HelperText mt="1">
                    {modelId === "base"
                        ? "Base model (chung, chưa tinh chỉnh)."
                        : "Fine-tuned model (đã tinh chỉnh tiếng Đức)."}
                </Field.HelperText>
            </Field.Root>
        </Box>
    );
}