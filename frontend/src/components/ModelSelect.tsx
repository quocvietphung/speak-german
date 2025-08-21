"use client";

import * as React from "react";
import {
  Field,
  NativeSelect,
  Box,
  HStack,
  Badge,
  useToken,
} from "@chakra-ui/react";

type ModelValue = "base" | "fine_tuned";

export interface ModelSelectProps {
  modelId: ModelValue;
  onChange: (value: ModelValue) => void;
}

export default function ModelSelect({ modelId, onChange }: ModelSelectProps) {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) =>
    onChange(e.currentTarget.value as ModelValue);

  const label =
    modelId === "base" ? "whisper_tiny_de" : "whisper_tiny_de_finetuned";

  // Pastel blue highlight for focus effect
  const pastelBlue = useToken("colors", "blue.200");

  return (
    <Box p="4" bg="blue.50" rounded="md" borderWidth="1px" borderColor="blue.100" w="100%">
      <Field.Root>
        <HStack justify="space-between" mb="2">
          <Field.Label fontSize="lg" fontWeight="semibold" color="blue.800">
            Select Model
          </Field.Label>
          <Badge variant="subtle" colorPalette="blue">
            {label}
          </Badge>
        </HStack>

        <NativeSelect.Root
          size="md"
          width="100%"
          bg="white"
          borderColor="blue.200"
          _focus={{
            borderColor: "blue.400",
            boxShadow: `0 0 0 1px ${pastelBlue}`,
          }}
        >
          <NativeSelect.Field
            value={modelId}
            onChange={handleChange}
            aria-label="Choose ASR model"
          >
            <option value="base">whisper_tiny_de</option>
            <option value="fine_tuned">whisper_tiny_de_finetuned</option>
          </NativeSelect.Field>
          <NativeSelect.Indicator />
        </NativeSelect.Root>

        <Field.HelperText mt="2" color="blue.600">
          {modelId === "base"
            ? "Using standard base model (not fine-tuned)."
            : "Using German fine-tuned model."}
        </Field.HelperText>
      </Field.Root>
    </Box>
  );
}