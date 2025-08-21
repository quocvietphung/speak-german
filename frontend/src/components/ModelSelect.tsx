"use client";

import * as React from "react";
import {
  Field,
  NativeSelect,
  Box,
  HStack,
  Badge,
} from "@chakra-ui/react";

type ModelValue = "base" | "fine_tuned";

export interface ModelSelectProps {
  modelId: ModelValue;
  onChange: (value: ModelValue) => void;
}

export default function ModelSelect({ modelId, onChange }: ModelSelectProps) {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onChange(e.currentTarget.value as ModelValue);
  };

  const label =
    modelId === "base" ? "whisper_tiny_de" : "whisper_tiny_de_finetuned";

  return (
    <Box
      bg="white"
      p={4}
      rounded="md"
      shadow="sm"
      border="1px solid"
      borderColor="gray.200"
    >
      <Field.Root>
        {/* Header with label + badge */}
        <HStack justify="space-between" mb="2">
          <Field.Label fontWeight="medium" color="gray.700">
            Select Model
          </Field.Label>
          <Badge colorPalette="blue" variant="subtle" rounded="md">
            {label}
          </Badge>
        </HStack>

        <NativeSelect.Root size="sm" width="50%">
          <NativeSelect.Field
            value={modelId}
            onChange={handleChange}
            aria-label="Select ASR model"
          >
            <option value="base">whisper_tiny_de</option>
            <option value="fine_tuned">whisper_tiny_de_finetuned</option>
          </NativeSelect.Field>
          <NativeSelect.Indicator />
        </NativeSelect.Root>

        {/* Helper text */}
        <Field.HelperText mt="2" color="gray.500" fontSize="sm">
          {modelId === "base"
            ? "Using standard base model (not fine-tuned)."
            : "Using fine-tuned German ASR model."}
        </Field.HelperText>
      </Field.Root>
    </Box>
  );
}